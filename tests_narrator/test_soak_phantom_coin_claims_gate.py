"""`phantom_coin_claim_gate` mirrors `phantom_hp_claim_gate`'s own shape exactly: it reads
`score_phantom_coin_claims`'s `phantom_coin_turns` field (merged into
`report.dice_visibility` by `run_soak`) and returns the pass/detail pair `run_soak`
feeds into `report.check(\"phantom-coin-claims\", ...)`, unconditionally -- see that call
site's own comment for why this check, like its `phantom-hp-claims` and
`phantom-roll-claims` siblings, does not sit behind `--require-continuity` or a probe
flag. This module needs no live model: both functions are pure, reading only the dicts
a run already produces. It lives beside the other soak-gate suites in
`tests_narrator/`, exercising `narrator.soak_instruments`.
"""

from __future__ import annotations

from narrator.soak_instruments import (
    SoakReport,
    phantom_coin_claim_gate,
    score_phantom_coin_claims,
    score_phantom_hp_claims,
    score_phantom_roll_claims,
    score_roll_announcements,
)


def _dice_visibility(tool_log: list[list[str]], posts: list[str]) -> dict:
    """The exact dict shape ``run_soak`` builds: ``score_roll_announcements``'s
    result, merged with its roll, hp, and coin phantom-claim siblings in turn,
    ``scorable`` set True.
    """
    visibility = score_roll_announcements(tool_log, posts)
    visibility.update(score_phantom_roll_claims(tool_log, posts))
    visibility.update(score_phantom_hp_claims(tool_log, posts))
    visibility.update(score_phantom_coin_claims(tool_log, posts))
    visibility["scorable"] = True
    return visibility


# --- phantom_coin_claim_gate, direct -------------------------------------------------


def test_phantom_coin_claim_gate_passes_when_every_scored_turn_is_clean():
    tool_log = [["character_sheet"], [], []]
    posts = [
        "Rill: 22 coins, a long knife, and a coil of rope.",
        "You look around the market.",
        "Nothing happens.",
    ]

    passed, detail = phantom_coin_claim_gate(_dice_visibility(tool_log, posts))

    assert passed is True
    assert detail == "0 phantom coin claim(s) among 3 scored turns"


def test_phantom_coin_claim_gate_names_an_unbacked_transfer():
    """Phantom coin claim gate names an unbacked transfer."""
    tool_log: list[list[str]] = [[] for _ in range(6)]
    posts = ["What do you do?" for _ in range(6)]
    posts[0] = "Three coins change hands, and a buckle is placed on the bench."
    posts[3] = 'The revised balance is 22 coins; the ledger will be corrected.'
    posts[5] = "You have 22 coins."

    passed, detail = phantom_coin_claim_gate(_dice_visibility(tool_log, posts))

    assert passed is False
    assert "2 phantom coin claim(s)" in detail
    assert "[4, 6]" in detail


def test_phantom_coin_claim_gate_is_zero_tolerance_not_a_partial_threshold():
    """One phantom claim among otherwise-clean turns still fails the run.

    The same zero-tolerance posture ``phantom_hp_claim_gate``/``phantom_roll_claim_gate``
    already take for a narrated figure with no matching tool call: a narrated coin
    figure with no matching tool request behind it is the identical defect class
    ../CONTRIBUTING.md's own invariant forbids, so this gate admits no quota either.
    """
    dice_visibility = {"phantom_coin_turns": [14], "turns_scored": 37}
    passed, _detail = phantom_coin_claim_gate(dice_visibility)
    assert passed is False


def test_phantom_coin_claim_gate_reads_missing_field_as_clean_not_failed():
    """A dict predating this fix (no ``phantom_coin_turns`` key) is read as zero.

    ``run_soak`` never calls this gate on such a dict in production (the unscorable
    branch fails ``phantom-coin-claims`` directly with its own detail, see that call
    site) -- this pins the function's own defensive default rather than a production
    path.
    """
    passed, detail = phantom_coin_claim_gate({"scorable": True})
    assert passed is True
    assert detail == "0 phantom coin claim(s) among 0 scored turns"


def test_phantom_coin_claim_gate_never_flags_the_fixed_trade_completed_notice():
    """The one real coin-mutating tool, ``engine_confirm_purchase``, is engine-only
    and never appears in ``tool_log``; the design that keeps this gate sound anyway is
    that its own delivered notice carries no digits at all. A soak run that completes
    a real purchase must never fail this gate on the turn the purchase itself lands.
    """
    tool_log = [[]]
    posts = ["The confirmed purchase was recorded."]

    passed, detail = phantom_coin_claim_gate(_dice_visibility(tool_log, posts))

    assert passed is True
    assert detail == "0 phantom coin claim(s) among 1 scored turns"


# --- exit-code proof: report.failed() differs, all other gates equal -----------------


def _report_with_unrelated_passing_checks() -> SoakReport:
    """A report carrying the same unrelated, passing checks in both scenarios below,
    mirroring ``test_soak_phantom_hp_claims_gate.py``'s own fixture: the gate's
    exit-code behaviour (``main`` returns ``0 if not report.failed() else 1``) must be
    proven nonzero on a session containing a phantom coin claim and zero otherwise,
    with every other gate equal.
    """
    report = SoakReport(seed=1, window_size=6, generated_messages=41)
    report.check("no-raw-leaks-delivered", True, "0 of 41 posted messages carried raw error text")
    report.check("withheld-posts-are-a-notice", True, "0 engine-notice posts for 0 withheld turns")
    return report


def test_report_failed_is_empty_when_no_phantom_coin_claim_all_other_gates_equal():
    tool_log = [["character_sheet"], [], []]
    posts = [
        "Rill: 22 coins, a long knife, and a coil of rope.",
        "You look around the market.",
        "Nothing happens.",
    ]
    report = _report_with_unrelated_passing_checks()
    report.dice_visibility = _dice_visibility(tool_log, posts)

    passed, detail = phantom_coin_claim_gate(report.dice_visibility)
    report.check("phantom-coin-claims", passed, detail)

    assert report.failed() == []


def test_report_failed_is_nonzero_when_a_phantom_coin_claim_is_present_all_other_gates_equal():
    """Identical to the clean scenario above except turn 2 now claims an unbacked figure."""
    tool_log = [["character_sheet"], [], []]
    posts = [
        "Rill: 22 coins, a long knife, and a coil of rope.",
        "You have 22 coins.",  # turn 2: no tool call backs this
        "Nothing happens.",
    ]
    report = _report_with_unrelated_passing_checks()
    report.dice_visibility = _dice_visibility(tool_log, posts)

    passed, detail = phantom_coin_claim_gate(report.dice_visibility)
    report.check("phantom-coin-claims", passed, detail)

    assert report.failed() == ["phantom-coin-claims"]
