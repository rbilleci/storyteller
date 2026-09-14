"""Pins scripts/probe_traversal.py's ``scenario_checks`` mapping and ``_leg_progress``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import probe_traversal as pt  # noqa: E402

# -- cliff-path (M7, unchanged) -----------------------------------------------


def test_cliff_path_checks_pass_on_a_clean_zero_replay_run():
    checks = pt.scenario_checks(
        "cliff-path", turns=12, errors=False, narration_count=12,
        exceeding_threshold_count=0,
    )
    assert checks == {
        "service_completed": True,
        "narration_delivered": True,
        "zero_replay": True,
    }


def test_cliff_path_zero_replay_fails_when_any_pair_exceeds_threshold():
    checks = pt.scenario_checks(
        "cliff-path", turns=12, errors=False, narration_count=12,
        exceeding_threshold_count=1,
    )
    assert checks["zero_replay"] is False
    assert checks["service_completed"] is True, "an unrelated dimension must not also fail"


def test_cliff_path_service_completed_fails_on_a_reported_error():
    checks = pt.scenario_checks(
        "cliff-path", turns=12, errors=True, narration_count=12,
        exceeding_threshold_count=0,
    )
    assert checks["service_completed"] is False


def test_cliff_path_narration_delivered_fails_when_every_turn_was_a_notice():
    checks = pt.scenario_checks(
        "cliff-path", turns=12, errors=False, narration_count=0,
        exceeding_threshold_count=0,
    )
    assert checks["narration_delivered"] is False


# -- _leg_progress: the shared helper causeway and switchback both score on --


def _entry(filled) -> dict:
    return {"location_id": "irrelevant", "clock_filled": filled, "clock_segments": None}


def test_leg_progress_reports_completion_and_a_clean_increasing_sequence():
    leg = [_entry(1), _entry(2), _entry(3)]
    turn, sequence = pt._leg_progress(leg, segments=3)
    assert turn == 3
    assert sequence == [1, 2, 3]
    assert pt._strictly_increasing(sequence) is True


def test_leg_progress_drops_a_leading_none_run_before_the_clock_existed():
    """A leg's first turn or two may run before any ``scene_commit`` opens the
    clock at all; that absence must not itself read as a monotonicity break.
    """
    leg = [_entry(None), _entry(None), _entry(1), _entry(2)]
    turn, sequence = pt._leg_progress(leg, segments=2)
    assert turn == 4
    assert sequence == [1, 2]


def test_leg_progress_reports_no_completion_when_the_budget_runs_out_first():
    """The budget can run out on a genuinely increasing but incomplete run --
    ``completion`` and ``_strictly_increasing`` are independent questions.
    """
    leg = [_entry(1), _entry(2)]
    turn, sequence = pt._leg_progress(leg, segments=3)
    assert turn is None
    assert sequence == [1, 2]
    assert pt._strictly_increasing(sequence) is True


def test_leg_progress_return_leg_drops_a_stale_prefix_before_the_reopen_lands():
    """The exact live shape this milestone's own evidence measured: the return
    leg's clock starts at the outbound leg's already-finished maximum for one
    turn before some later call actually reopens it. Without
    ``allow_stale_start``, that leading run must not itself be mistaken for
    instant completion or a strictly-increasing run of one flat value.
    """
    leg = [_entry(3), _entry(0), _entry(1), _entry(2), _entry(3)]
    turn, sequence = pt._leg_progress(leg, segments=3, allow_stale_start=True)
    assert turn == 5, "the stale turn still counts against this leg's own budget"
    assert sequence == [0, 1, 2, 3]
    assert pt._strictly_increasing(sequence) is True


def test_leg_progress_return_leg_reports_no_completion_when_never_reopened():
    """A return leg whose clock never actually resets (the model only ever
    re-declares against the stale maximum) must not read as instantly
    complete on turn one -- that would credit a leg that made zero real
    progress of its own.
    """
    leg = [_entry(3), _entry(3), _entry(3)]
    turn, sequence = pt._leg_progress(leg, segments=3, allow_stale_start=True)
    assert turn is None
    assert sequence == []


def test_leg_progress_without_allow_stale_start_reads_an_already_full_clock_as_instant():
    """The outbound leg's own contract: unlike the return leg, an outbound
    clock reading ``segments`` on its very first observed turn is a real,
    immediate completion (a short crossing, or a model that opened and
    finished it in one call) and must read that way, not be second-guessed as
    a stale carryover -- carryover only ever applies to a *reused* clock,
    which ``allow_stale_start`` names explicitly.
    """
    leg = [_entry(3)]
    turn, sequence = pt._leg_progress(leg, segments=3)
    assert turn == 1
    assert sequence == [3]


def test_strictly_increasing_rejects_a_flat_repeat_and_a_decrease():
    assert pt._strictly_increasing([1, 2, 2]) is False
    assert pt._strictly_increasing([2, 1]) is False
    assert pt._strictly_increasing([]) is False
    assert pt._strictly_increasing([1, 2, 3]) is True


def _causeway_leg(fills: list[int | None]) -> list[dict]:
    return [_entry(value) for value in fills]


def test_causeway_checks_pass_on_a_clean_symmetric_crossing():
    checks = pt.scenario_checks(
        "causeway", turns=6, errors=False, narration_count=6,
        exceeding_threshold_count=0,
        outbound_log=_causeway_leg([1, 2, 3]),
        return_log=_causeway_leg([1, 2, 3]),
        segments=3,
    )
    assert checks == {
        "service_completed": True,
        "narration_delivered": True,
        "outbound_completed": True,
        "outbound_within_bound": True,
        "outbound_fill_strictly_increasing": True,
        "return_completed": True,
        "return_within_bound": True,
        "return_fill_strictly_increasing": True,
        "turn_counts_within_one": True,
    }


def test_causeway_within_bound_fails_past_segments_plus_one_turns():
    """3 segments -> bound is 4; a stalled turn pushes completion to turn 5,
    one past the bound."""
    checks = pt.scenario_checks(
        "causeway", turns=10, errors=False, narration_count=10,
        exceeding_threshold_count=0,
        outbound_log=_causeway_leg([1, 2, 2, 2, 3]),
        return_log=_causeway_leg([1, 2, 3]),
        segments=3,
    )
    assert checks["outbound_completed"] is True
    assert checks["outbound_within_bound"] is False


def test_causeway_turn_counts_within_one_fails_on_a_lopsided_crossing():
    checks = pt.scenario_checks(
        "causeway", turns=10, errors=False, narration_count=10,
        exceeding_threshold_count=0,
        outbound_log=_causeway_leg([1, 2, 3]),
        return_log=_causeway_leg([1, 1, 1, 2, 3]),
        segments=3,
    )
    assert checks["outbound_completed"] is True
    assert checks["return_completed"] is True
    assert checks["turn_counts_within_one"] is False, "3 vs 5 turns differ by 2"


def test_causeway_return_never_reopened_fails_return_completed_only():
    """The exact live gap this milestone's evidence surfaced and repaired
    (``clock_updates`` against an already-full traversal clock now restarts
    it, src/bsh_mcp/service.py): before that repair, a model representing the
    return trip only through ``clock_updates`` against the stale maximum left
    the clock pinned, never reopening. Pinned here as the scoring-side
    regression guard: if that repair ever regresses, this scenario's own
    checks must fail exactly this way, not silently pass.
    """
    checks = pt.scenario_checks(
        "causeway", turns=7, errors=False, narration_count=7,
        exceeding_threshold_count=0,
        outbound_log=_causeway_leg([1, 2, 3]),
        return_log=_causeway_leg([3, 3, 3, 3]),
        segments=3,
    )
    assert checks["outbound_completed"] is True
    assert checks["return_completed"] is False
    assert checks["return_within_bound"] is False
    assert checks["return_fill_strictly_increasing"] is False
    assert checks["turn_counts_within_one"] is False


def test_causeway_outbound_never_completing_fails_service_independent_checks():
    checks = pt.scenario_checks(
        "causeway", turns=4, errors=False, narration_count=4,
        exceeding_threshold_count=0,
        outbound_log=_causeway_leg([1, 1, 1, 1]),
        return_log=[],
        segments=3,
    )
    assert checks["service_completed"] is True
    assert checks["outbound_completed"] is False
    assert checks["outbound_within_bound"] is False
    assert checks["outbound_fill_strictly_increasing"] is False
    assert checks["return_completed"] is False


def test_switchback_checks_pass_on_a_clean_six_segment_crossing():
    checks = pt.scenario_checks(
        "switchback", turns=6, errors=False, narration_count=6,
        exceeding_threshold_count=0,
        outbound_log=_causeway_leg([1, 2, 3, 4, 5, 6]),
        segments=6,
    )
    assert checks == {
        "service_completed": True,
        "narration_delivered": True,
        "outbound_completed": True,
        "outbound_within_bound": True,
        "outbound_fill_strictly_increasing": True,
        "fill_at_least_turn_index": True,
        "no_replay_above_threshold": True,
    }


def test_switchback_fill_at_least_turn_index_fails_when_a_turn_lags_behind():
    """fill after k declarations >= k, per turn -- a turn that lags (fill 2
    after 3 declarations) must fail even though the run eventually completes.
    """
    checks = pt.scenario_checks(
        "switchback", turns=7, errors=False, narration_count=7,
        exceeding_threshold_count=0,
        outbound_log=_causeway_leg([1, 1, 2, 3, 4, 5, 6]),
        segments=6,
    )
    assert checks["outbound_completed"] is True
    assert checks["fill_at_least_turn_index"] is False, "turn 2 read fill 1, below its own index"


def test_switchback_no_replay_above_threshold_reads_the_exceeding_count():
    checks = pt.scenario_checks(
        "switchback", turns=7, errors=False, narration_count=7,
        exceeding_threshold_count=2,
        outbound_log=_causeway_leg([1, 2, 3, 4, 5, 6]),
        segments=6,
    )
    assert checks["no_replay_above_threshold"] is False


def test_switchback_within_bound_fails_past_seven_declarations_for_six_segments():
    """6 segments -> bound is 7; a stalled turn pushes completion to turn 8,
    one past the bound."""
    checks = pt.scenario_checks(
        "switchback", turns=8, errors=False, narration_count=8,
        exceeding_threshold_count=0,
        outbound_log=_causeway_leg([1, 2, 3, 4, 5, 5, 5, 6]),
        segments=6,
    )
    assert checks["outbound_completed"] is True
    assert checks["outbound_within_bound"] is False, "completion landed on turn 8, one past the bound of 7"
