"""Pin the promised-unbound-check detector and its escalation.

This file needs no live model and no campaign on disk: `_plan_log_entry` is a
pure function of a `PlanOutcome | DeclinedPlan`, and `NarratorEngine(config)`
constructs without `engine.start()`, the same posture
`tests_narrator/test_engine_uncommitted_narration.py` already documents. It needs
`strands` (`narrator.engine` imports it transitively), the reason sibling
`tests_narrator/test_engine_*.py` files live here rather than under `tests/`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fake_classifier import fake_classify_intent

from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.config import NarratorConfig
from narrator.decisions import (
    ApproachDraft,
    ApproachOption,
    AttributeTestProposal,
    ChoiceOption,
    ClarificationDraft,
    ConfirmationDraft,
    CurrentAudience,
    DeclinedPlan,
    NoTestProposal,
    PartyAudience,
    PlanOutcome,
    ProceedPlan,
    RequestDecisionPlan,
)
from narrator.engine import PROMISED_CHECK_PATTERN, NarratorEngine, _plan_log_entry


def _clarification(*, question: str = "How do you proceed?", context: str = "", labels=()) -> PlanOutcome:
    paths = tuple(
        ChoiceOption(id=f"path_{i}", label=label) for i, label in enumerate(labels or ("Do something.",))
    )
    draft = ClarificationDraft(
        kind="clarification", question=question, context=context,
        audience=CurrentAudience(kind="current"), paths=paths,
    )
    return PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=draft))


def _confirmation(*, question: str = "Confirm?", context: str = "") -> PlanOutcome:
    draft = ConfirmationDraft(
        kind="confirmation", question=question, context=context, audience=CurrentAudience(kind="current"),
    )
    return PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=draft))


def _approach(*, question: str = "Choose your approach.", context: str = "") -> PlanOutcome:
    draft = ApproachDraft(
        kind="approach", question=question, context=context, audience=PartyAudience(kind="party"),
        approaches=(
            ApproachOption(
                id="climb", label="Climb using Strength (requires a check).",
                directive=AttributeTestProposal(kind="attribute_test", attribute="STR"),
            ),
        ),
    )
    return PlanOutcome(plan=RequestDecisionPlan(kind="request", decision=draft))


# -- PROMISED_CHECK_PATTERN, direct ----------------------------------------------


def test_the_pattern_matches_synthetic_promised_checks():
    assert PROMISED_CHECK_PATTERN.search('Mend the broken cart axle (requires a check).')
    assert PROMISED_CHECK_PATTERN.search("Inspect the faded map (requires WIS/INT check).")
    assert PROMISED_CHECK_PATTERN.search("Lift the stone lid (Strength-based approach, requires a roll).")


def test_the_pattern_does_not_match_ordinary_decision_text():
    assert not PROMISED_CHECK_PATTERN.search("Attempt a standard climb using only the handholds.")
    assert not PROMISED_CHECK_PATTERN.search("Search the nearby stalls for a rope first.")


# -- _plan_log_entry ---------------------------------------------------------------


def test_proceed_and_declined_are_never_flagged():
    assert _plan_log_entry(PlanOutcome(plan=ProceedPlan(kind="proceed"))) == {
        "kind": "proceed", "promised_unbound_check": False,
    }
    assert _plan_log_entry(DeclinedPlan(kind="declined")) == {
        "kind": "declined", "promised_unbound_check": False,
    }


def test_a_clarification_with_no_promise_is_not_flagged():
    entry = _plan_log_entry(_clarification(labels=("Attempt a standard climb.", "Search for equipment first.")))
    assert entry == {"kind": "clarification", "promised_unbound_check": False}


def test_a_clarification_promising_a_check_in_an_option_label_is_flagged():
    """Test a clarification promising a check in an option label is flagged.
    """
    entry = _plan_log_entry(
        _clarification(
            question='What is your next step?',
            labels=('Mend the broken cart axle (requires a check).', "Pick it up as-is."),
        )
    )
    assert entry == {"kind": "clarification", "promised_unbound_check": True}


def test_a_clarification_promising_a_check_in_its_own_question_is_flagged():
    entry = _plan_log_entry(_clarification(question="Reading this dim script requires a check. How do you attempt it?"))
    assert entry == {"kind": "clarification", "promised_unbound_check": True}


def test_a_clarification_promising_a_check_in_its_own_context_is_flagged():
    entry = _plan_log_entry(_clarification(context="Forcing this door requires a roll to succeed."))
    assert entry == {"kind": "clarification", "promised_unbound_check": True}


def test_a_confirmation_promising_a_check_is_flagged():
    entry = _plan_log_entry(_confirmation(context="This forced entry requires an Athletics check."))
    assert entry == {"kind": "confirmation", "promised_unbound_check": True}


def test_an_approach_promising_a_check_is_reported_but_never_flagged():
    """The directive-bound shape is exactly what production already enforces
    (`resolution_guard.ResolutionGuard`): its own option text can say "requires a
    check" freely, because unlike a clarification it actually carries a
    ``ResolutionDirective`` (``ApproachOption.directive``) something downstream
    can hold it to.
    """
    entry = _plan_log_entry(_approach())
    assert entry == {"kind": "approach", "promised_unbound_check": False}


# -- _escalate_promised_check, direct -----------------------------------------------


def _engine(tmp_path: Path) -> NarratorEngine:
    """An unstarted engine that can still route a turn offline.

    ``plan_turn`` classifies before every hazard decision, and an unstarted engine's
    own ``classify_intent`` answers ``None`` so it reaches no endpoint. ``None`` is
    the fail-closed value, so without a stand-in every declaration here would draw the
    uncategorised confirmation instead of the plan under test. The double supplies
    deterministic verdicts; see ``tests_narrator/lexical_double.py`` for what it does
    and does not establish.
    """
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    engine.classify_intent = fake_classify_intent
    return engine


def test_a_bound_directive_is_surfaced_as_the_replacement(tmp_path: Path, monkeypatch):
    engine = _engine(tmp_path)
    draft = ApproachDraft(
        kind="approach", question="How?", audience=PartyAudience(kind="party"),
        approaches=(
            ApproachOption(
                id="force", label="Force it (STR).",
                directive=AttributeTestProposal(kind="attribute_test", attribute="STR"),
            ),
        ),
    )

    async def fake_escalate_once(declaration, canon_text):
        return draft

    monkeypatch.setattr(engine, "_escalate_once", fake_escalate_once)
    replacement, disposition = asyncio.run(engine._escalate_promised_check("force the door", "canon"))

    assert disposition == "bound"
    assert isinstance(replacement, PlanOutcome)
    assert isinstance(replacement.plan, RequestDecisionPlan)
    assert replacement.plan.decision is draft


def test_every_option_no_test_collapses_to_a_plain_proceed(tmp_path: Path, monkeypatch):
    """The model retracting its own first-pass promise on reflection: no
    player-facing decision spent on a check nobody, on a closer look, needed."""
    engine = _engine(tmp_path)
    draft = ApproachDraft(
        kind="approach", question="How?", audience=PartyAudience(kind="party"),
        approaches=(
            ApproachOption(id="just_open", label="Just open it.", directive=NoTestProposal(kind="no_test")),
        ),
    )

    async def fake_escalate_once(declaration, canon_text):
        return draft

    monkeypatch.setattr(engine, "_escalate_once", fake_escalate_once)
    replacement, disposition = asyncio.run(engine._escalate_promised_check("open the door", "canon"))

    assert disposition == "collapsed"
    assert replacement == PlanOutcome(plan=ProceedPlan(kind="proceed"))


def test_a_mixed_reply_with_any_real_directive_still_binds(tmp_path: Path, monkeypatch):
    """One genuine option among several ``no_test`` ones is still worth surfacing:
    the player may prefer the riskier-but-faster path, so a single bound
    directive is enough to bind, not a requirement that every option agree."""
    engine = _engine(tmp_path)
    draft = ApproachDraft(
        kind="approach", question="How?", audience=PartyAudience(kind="party"),
        approaches=(
            ApproachOption(id="pry", label="Pry it open quietly.", directive=NoTestProposal(kind="no_test")),
            ApproachOption(
                id="force", label="Force it (STR).",
                directive=AttributeTestProposal(kind="attribute_test", attribute="STR"),
            ),
        ),
    )

    async def fake_escalate_once(declaration, canon_text):
        return draft

    monkeypatch.setattr(engine, "_escalate_once", fake_escalate_once)
    replacement, disposition = asyncio.run(engine._escalate_promised_check("force the door", "canon"))

    assert disposition == "bound"
    assert replacement.plan.decision is draft


def test_escalation_failure_is_fail_open_not_propagated(tmp_path: Path, monkeypatch):
    """Repairs fiction-level text, never durable state: a broken escalation must
    never cost the turn anything, matching ``_sweep``'s own fail-open contract."""
    engine = _engine(tmp_path)

    async def fake_escalate_once(declaration, canon_text):
        raise RuntimeError("endpoint hiccup")

    monkeypatch.setattr(engine, "_escalate_once", fake_escalate_once)
    replacement, disposition = asyncio.run(engine._escalate_promised_check("force the door", "canon"))

    assert replacement is None
    assert disposition == "failed"


def test_a_none_reply_from_escalate_once_is_also_failed(tmp_path: Path, monkeypatch):
    engine = _engine(tmp_path)

    async def fake_escalate_once(declaration, canon_text):
        return None

    monkeypatch.setattr(engine, "_escalate_once", fake_escalate_once)
    replacement, disposition = asyncio.run(engine._escalate_promised_check("force the door", "canon"))

    assert replacement is None
    assert disposition == "failed"


# -- plan_turn wiring ----------------------------------------------------------------


def _plan_turn_with_stubbed_planner(tmp_path: Path, monkeypatch, first_pass, escalate_result):
    """Drive the real ``plan_turn`` with ``_plan_once``/``_escalate_once`` stubbed,
    the same seam ``tests_narrator/test_engine_uncommitted_narration.py`` already
    uses for ``_plan_once`` alone.
    """
    engine = _engine(tmp_path)

    async def fake_plan_once(prompt, *, repair=False):
        return first_pass.model_dump()

    async def fake_escalate_once(declaration, canon_text):
        return escalate_result

    monkeypatch.setattr(engine, "_plan_once", fake_plan_once)
    monkeypatch.setattr(engine, "_escalate_once", fake_escalate_once)

    turn = InboundTurn("probe", ChannelMessage("Rill", "I try to force the door open."))
    result = asyncio.run(
        engine.plan_turn(turn, eligible_characters=(("rill", "Rill"),), prior_resolutions=())
    )
    return engine, result


def test_plan_turn_surfaces_a_bound_escalation_in_place_of_the_promised_clarification(tmp_path: Path, monkeypatch):
    first_pass = _clarification(
        question="How do you force it?",
        labels=("Force it with strength (requires a check).", 'Search for a different entrance.'),
    )
    bound = ApproachDraft(
        kind="approach", question="How?", audience=PartyAudience(kind="party"),
        approaches=(
            ApproachOption(
                id="force", label="Force it (STR).",
                directive=AttributeTestProposal(kind="attribute_test", attribute="STR"),
            ),
        ),
    )
    engine, result = _plan_turn_with_stubbed_planner(tmp_path, monkeypatch, first_pass, bound)

    assert isinstance(result.plan, RequestDecisionPlan)
    assert result.plan.decision.kind == "approach"
    assert engine._plan_log[-1]["kind"] == "clarification"
    assert engine._plan_log[-1]["promised_unbound_check"] is True
    assert engine._plan_log[-1]["escalation"] == "bound"
    assert engine._plan_log[-1]["final_kind"] == "approach"


def test_plan_turn_collapses_to_proceed_when_escalation_finds_no_test_needed(tmp_path: Path, monkeypatch):
    first_pass = _confirmation(context="This forced entry requires an Athletics check.")
    all_no_test = ApproachDraft(
        kind="approach", question="How?", audience=PartyAudience(kind="party"),
        approaches=(
            ApproachOption(id="just_open", label="Just open it.", directive=NoTestProposal(kind="no_test")),
        ),
    )
    engine, result = _plan_turn_with_stubbed_planner(tmp_path, monkeypatch, first_pass, all_no_test)

    assert isinstance(result.plan, ProceedPlan)
    assert engine._plan_log[-1]["escalation"] == "collapsed"
    assert engine._plan_log[-1]["final_kind"] == "proceed"


def test_plan_turn_keeps_the_original_decision_when_escalation_fails(tmp_path: Path, monkeypatch):
    first_pass = _clarification(labels=("Force it with strength (requires a check).",))
    engine, result = _plan_turn_with_stubbed_planner(tmp_path, monkeypatch, first_pass, None)

    assert isinstance(result.plan, RequestDecisionPlan)
    assert result.plan.decision.kind == "clarification"
    assert engine._plan_log[-1]["escalation"] == "failed"
    assert engine._plan_log[-1]["final_kind"] == "clarification"


def test_plan_turn_never_escalates_when_nothing_was_promised(tmp_path: Path, monkeypatch):
    """No wasted call: escalation only fires when the first pass actually
    promised a check with nothing to enforce it. ``_escalate_once`` is stubbed
    to raise if it is ever reached at all, so this fails loudly rather than
    silently passing if that guard regresses.
    """
    first_pass = _clarification(labels=("Attempt a standard climb.", "Search for equipment first."))

    async def unexpected_escalate_once(declaration, canon_text):
        raise AssertionError("must not escalate when nothing was promised")

    engine = _engine(tmp_path)

    async def fake_plan_once(prompt, *, repair=False):
        return first_pass.model_dump()

    monkeypatch.setattr(engine, "_plan_once", fake_plan_once)
    monkeypatch.setattr(engine, "_escalate_once", unexpected_escalate_once)

    turn = InboundTurn("probe", ChannelMessage("Rill", "I try to climb the wall."))
    result = asyncio.run(
        engine.plan_turn(turn, eligible_characters=(("rill", "Rill"),), prior_resolutions=())
    )

    assert isinstance(result.plan, RequestDecisionPlan)
    assert result.plan.decision.kind == "clarification"
    assert engine._plan_log[-1]["escalation"] == "not_triggered"
    assert engine._plan_log[-1]["final_kind"] == engine._plan_log[-1]["kind"]
