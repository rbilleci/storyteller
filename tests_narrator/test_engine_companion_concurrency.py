"""Pin C4: the two AFTER companions (hazard, route) dispatch concurrently when both
apply, instead of one after the other.

Needs `strands` (`narrator.engine` imports it transitively), the reason sibling
`tests_narrator/test_engine_*.py` files live here rather than under `tests/`. No live
model and no campaign on disk: `NarratorEngine(config)` constructs without
`engine.start()`, the same posture `test_engine_promised_check_detection.py` documents.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from narrator.classify import DEFAULT_TURN_CLASSIFIER, HAZARD_COMPANION, ROUTE_COMPANION
from narrator.config import NarratorConfig
from narrator.engine import NarratorEngine

_COMPANION_DELAY = 0.05


def _full_verdict(**overrides):
    fields = {
        "route": "read", "hazard": "none", "social_category": "none", "social_mode": "none",
        "trade_phase": "none", "named_person_ids": ["clerk"], "names_unrecorded_person": False,
        "interlocutor_id": "", "accepts_offer": True, "offered_price": 0, "departs": False,
        "bare_answer": "none", "replies_to_question": False, "defence_method": "none",
        "reads_by_observing": False, "romance_escalation": False, "romance_coercive": False,
        "reason": "probe",
    }
    fields.update(overrides)
    return DEFAULT_TURN_CLASSIFIER.schema(**fields)


def _hazard_companion_result(**overrides):
    fields = {
        "route": "read", "hazard": "theft", "named_person_ids": ["clerk"],
        "names_unrecorded_person": False, "interlocutor_id": "", "accepts_offer": True,
        "offered_price": 0, "reason": "probe",
    }
    fields.update(overrides)
    return HAZARD_COMPANION.schema(**fields)


def _route_companion_result(**overrides):
    fields = {"route": "read", "also_declares_act": True, "declared_act_kind": "watch", "reason": "probe"}
    fields.update(overrides)
    return ROUTE_COMPANION.schema(**fields)


def _engine(tmp_path: Path) -> NarratorEngine:
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    engine._client = object()  # noqa: SLF001 - started-engine seam; only the companion calls are stubbed
    return engine


async def test_both_after_companions_dispatch_concurrently_not_sequentially(tmp_path):
    """The behavior C4 changes: two ~0.05s stubbed companion calls, both applicable,
    must together cost close to one call's duration, not their sum."""
    engine = _engine(tmp_path)

    async def fake_classify_once(prompt):
        return _full_verdict()

    started: dict[str, float] = {}

    async def fake_companion_once(classifier, declaration, context, *, origin):
        key = "hazard" if classifier is engine.hazard_companion else "route"
        started[key] = time.perf_counter()
        await asyncio.sleep(_COMPANION_DELAY)
        return _hazard_companion_result() if key == "hazard" else _route_companion_result()

    engine._classify_once = fake_classify_once  # noqa: SLF001
    engine._companion_once = fake_companion_once  # noqa: SLF001

    start = time.perf_counter()
    verdict = await engine.classify_intent("Wait -- is that the ring the merchant lost? I'll take it.")
    elapsed = time.perf_counter() - start

    assert set(started) == {"hazard", "route"}, "both companions must have been asked"
    # Both start times land inside one call's own delay window of each other -- proof
    # they were dispatched together, not one awaited before the other began.
    assert abs(started["hazard"] - started["route"]) < _COMPANION_DELAY
    # Sequential dispatch would cost roughly 2x _COMPANION_DELAY; concurrent dispatch
    # costs roughly 1x. The threshold sits between the two with margin either way.
    assert elapsed < _COMPANION_DELAY * 1.6, f"took {elapsed:.3f}s, expected well under {_COMPANION_DELAY * 2:.3f}s"

    assert verdict.hazard == "theft"
    assert verdict.route == "read"
    assert verdict.also_declares_act is True
    assert verdict.declared_act_kind == "watch"


async def test_both_after_companions_merge_the_same_as_sequential_dispatch(tmp_path):
    """Correctness, not just speed: the merged verdict from concurrent dispatch must
    equal what today's sequential loop would have produced -- same fields, same
    values, regardless of which companion's task happens to finish first."""
    engine = _engine(tmp_path)

    async def fake_classify_once(prompt):
        return _full_verdict()

    # The route companion resolves first, the hazard companion second -- the reverse
    # of AFTER_COMPANIONS' own tuple order -- to prove merge order (not finish order)
    # decides the outcome, matching this module's own field-disjoint argument.
    async def fake_companion_once(classifier, declaration, context, *, origin):
        if classifier is engine.route_companion:
            return _route_companion_result()
        await asyncio.sleep(0.01)
        return _hazard_companion_result()

    engine._classify_once = fake_classify_once  # noqa: SLF001
    engine._companion_once = fake_companion_once  # noqa: SLF001

    verdict = await engine.classify_intent("Wait -- is that the ring the merchant lost? I'll take it.")

    assert verdict.hazard == "theft"
    assert verdict.also_declares_act is True
    assert verdict.declared_act_kind == "watch"
    assert engine._companion_log[-1]["hazard"] == "added"  # noqa: SLF001
    assert engine._companion_log[-1]["route"] == "added"  # noqa: SLF001


async def test_a_faulted_after_companion_does_not_block_the_other(tmp_path):
    """One companion faulting (``_companion_once``'s own contract: never raises,
    answers ``None``) must not prevent the other's result from merging -- matches
    the sequential loop's own per-companion fail-open reading."""
    engine = _engine(tmp_path)

    async def fake_classify_once(prompt):
        return _full_verdict()

    async def fake_companion_once(classifier, declaration, context, *, origin):
        if classifier is engine.hazard_companion:
            return None  # _companion_once's own fault value; it never raises
        return _route_companion_result()

    engine._classify_once = fake_classify_once  # noqa: SLF001
    engine._companion_once = fake_companion_once  # noqa: SLF001

    verdict = await engine.classify_intent("Wait -- is that the ring the merchant lost? I'll take it.")

    assert verdict.hazard == "none"  # unmerged: the hazard companion faulted
    assert verdict.also_declares_act is True  # route still merged
    assert engine._companion_log[-1]["hazard"] == "fault"  # noqa: SLF001
    assert engine._companion_log[-1]["route"] == "added"  # noqa: SLF001
