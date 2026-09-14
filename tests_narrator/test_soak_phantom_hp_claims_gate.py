"""Pin the M21 fix: a phantom hit-point claim fails the soak gate's exit code.

`phantom_hp_claim_gate` is the pure function this fix adds, mirroring
`phantom_roll_claim_gate`'s own shape exactly: it reads `score_phantom_hp_claims`'s
`phantom_hp_turns` field (merged into `report.dice_visibility` by `run_soak`) and
returns the pass/detail pair `run_soak` feeds into
`report.check(\"phantom-hp-claims\", ...)`, unconditionally -- see that call site's own
comment for why this check, like its `phantom-roll-claims` sibling, does not sit behind
`--require-continuity` or a probe flag. This module needs no live model: both functions
are pure, reading only the dicts a run already produces. It lives beside the other
soak-gate suites in `tests_narrator/`, exercising `narrator.soak_instruments`.
"""

from __future__ import annotations

from narrator.soak_instruments import (
    SoakReport,
    phantom_hp_claim_gate,
    score_phantom_hp_claims,
    score_phantom_roll_claims,
    score_roll_announcements,
)


def _dice_visibility(tool_log: list[list[str]], posts: list[str]) -> dict:
    """The exact dict shape ``run_soak`` builds: ``score_roll_announcements``'s
    result, merged with ``score_phantom_roll_claims``'s and then
    ``score_phantom_hp_claims``'s, ``scorable`` set True.
    """
    visibility = score_roll_announcements(tool_log, posts)
    visibility.update(score_phantom_roll_claims(tool_log, posts))
    visibility.update(score_phantom_hp_claims(tool_log, posts))
    visibility["scorable"] = True
    return visibility


# --- phantom_hp_claim_gate, direct ---------------------------------------------------


def test_phantom_hp_claim_gate_passes_when_every_scored_turn_is_clean():
    tool_log = [["rest"], [], []]
    posts = [
        "Ossa recovers 5 hit points (12/15).",
        "You look around the market.",
        "Nothing happens.",
    ]

    passed, detail = phantom_hp_claim_gate(_dice_visibility(tool_log, posts))

    assert passed is True
    assert detail == "0 phantom hit-point claim(s) among 3 scored turns"


def test_phantom_hp_claim_gate_fails_and_names_the_turn_seed_2026_shape():
    """Same fixture ``tests_narrator/test_soak_offline_instruments.py``'s
    ``test_score_phantom_hp_claims_detects_unbacked_recovery`` pins for
    the detector alone; here the full gate pass/detail pair is checked.
    """
    tool_log: list[list[str]] = [[] for _ in range(37)]
    posts = ["An empty cart rattles past." for _ in range(37)]
    posts[4] = "Ossa recovers 5 hit points."
    posts[10] = "Ossa recovers 5 hit points."

    passed, detail = phantom_hp_claim_gate(_dice_visibility(tool_log, posts))

    assert passed is False
    assert "2 phantom hit-point claim(s)" in detail
    assert "[5, 11]" in detail


def test_phantom_hp_claim_gate_is_zero_tolerance_not_a_partial_threshold():
    """One phantom claim among otherwise-clean turns still fails the run.

    The same zero-tolerance posture `phantom_roll_claim_gate` already takes for a
    narrated roll verdict with no matching tool call: a narrated hit-point figure
    with no matching tool request behind it is the identical defect class
    ../CONTRIBUTING.md's own invariant forbids, so this gate admits no quota either.
    """
    dice_visibility = {"phantom_hp_turns": [14], "turns_scored": 37}
    passed, _detail = phantom_hp_claim_gate(dice_visibility)
    assert passed is False


def test_phantom_hp_claim_gate_reads_missing_field_as_clean_not_failed():
    """A dict predating this fix (no ``phantom_hp_turns`` key) is read as zero.

    ``run_soak`` never calls this gate on such a dict in production (the unscorable
    branch fails ``phantom-hp-claims`` directly with its own detail, see that call
    site) -- this pins the function's own defensive default rather than a
    production path.
    """
    passed, detail = phantom_hp_claim_gate({"scorable": True})
    assert passed is True
    assert detail == "0 phantom hit-point claim(s) among 0 scored turns"


# --- exit-code proof: report.failed() differs, all other gates equal -----------------


def _report_with_unrelated_passing_checks() -> SoakReport:
    """A report carrying the same unrelated, passing checks in both scenarios below.
    """
    report = SoakReport(seed=1, window_size=6, generated_messages=41)
    report.check("no-raw-leaks-delivered", True, "0 of 41 posted messages carried raw error text")
    report.check("withheld-posts-are-a-notice", True, "0 engine-notice posts for 0 withheld turns")
    return report


def test_report_failed_is_empty_when_no_phantom_hp_claim_all_other_gates_equal():
    tool_log = [["rest"], [], []]
    posts = [
        "Ossa recovers 5 hit points (12/15).",
        "You look around the market.",
        "Nothing happens.",
    ]
    report = _report_with_unrelated_passing_checks()
    report.dice_visibility = _dice_visibility(tool_log, posts)

    passed, detail = phantom_hp_claim_gate(report.dice_visibility)
    report.check("phantom-hp-claims", passed, detail)

    assert report.failed() == []


def test_report_failed_is_nonzero_when_a_phantom_hp_claim_is_present_all_other_gates_equal():
    """Identical to the clean scenario above except turn 2 now claims an unbacked figure.

    ``report.failed()`` feeding ``main``'s ``return 0 if not failed else 1`` is what
    actually decides the soak run's exit code, so this is the exit-code proof this
    milestone's own acceptance criterion asks for, not only the gate function's own
    return value.
    """
    tool_log = [["rest"], [], []]
    posts = [
        "Ossa recovers 5 hit points (12/15).",
        "Rill recovers 3 hit points.",  # turn 2: no tool call backs this
        "Nothing happens.",
    ]
    report = _report_with_unrelated_passing_checks()
    report.dice_visibility = _dice_visibility(tool_log, posts)

    passed, detail = phantom_hp_claim_gate(report.dice_visibility)
    report.check("phantom-hp-claims", passed, detail)

    assert report.failed() == ["phantom-hp-claims"]
