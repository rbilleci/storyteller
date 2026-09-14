"""Classifier breadth checks over synthetic and generic regression inputs.

The multilingual corpus is the strict safety gate. This companion probe measures
agreement with the lexical double, with named exclusions for known semantic
disagreements. Changed inputs have no inherited live acceptance result.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from regression_corpus import REGRESSION_CASES  # noqa: E402

from narrator.classify import policy_from  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402
from narrator.engine import NarratorEngine  # noqa: E402
from narrator.policy_types import CombatSnapshot, TrustedScope  # noqa: E402

#: The three scope labels the harvest recorded often enough to reconstruct exactly.
#: Cases outside them carry a bespoke scope expression in their ``scope`` field; they
#: are skipped and counted, never silently dropped -- ``test_the_skipped_cases_are_reported``
#: asserts the count is visible.
_SCOPES: dict[str, tuple[TrustedScope, CombatSnapshot | None]] = {
    "no_one_present": (TrustedScope("session-1", "the-eel-market", ()), None),
    "npc_present": (
        TrustedScope(
            "session-1", "the-eel-market", ("orso-pell",), ("rill",),
            (("orso-pell", "alive"),),
        ),
        None,
    ),
    "fight": (
        TrustedScope(
            "session-1", "the-eel-market", ("choir-listener",), ("rill",),
            (("choir-listener", "alive"),),
        ),
        CombatSnapshot(active=True, round=1, sides=(("choir-listener", "npc"),)),
    ),
}


_DELIBERATE_DISAGREEMENTS: dict[str, str] = {
    "take the queen from the rusted cabinet": (
        "A chess piece in a cabinet is not a person. The lexicon needed "
        "_AMBIGUOUS_PERSON_WORDS and _CHESS_CONTEXT_NOUNS to reach the same answer and "
        "still flagged this one; the classifier reads the cabinet."
    ),
    "reed thug is punching me": (
        "The NPC is the actor. The player declares nothing, so no consent ask is owed. "
        "The lexicon matched an inflection of a violent root without asking who acts."
    ),
    "I killed reed thug": (
        "A report of a past event is not a new declaration of violence."
    ),
    "I harmed reed thug": "Past tense report; same as 'I killed reed thug'.",
    "Should I open the door, then I stab reed thug?": (
        "Asks permission rather than declaring. AUD-3 made the lexicon commit any "
        "'then'-chained clause regardless of the question opener governing it."
    ),
    "What if I open the door, then I take his purse?": (
        "A hypothetical, for the same reason as the case above."
    ),
}

#: Agreement floor on cases carrying a positive label. Set from the measured baseline
#: with margin rather than from an aspiration: a floor above what the classifier
#: actually scores would fail on arrival, and one at 100 percent would demand it
#: reproduce the six defects ``declaration_corpus`` pins as fixed.
_AGREEMENT_FLOOR = 0.75


def _runnable() -> list[dict]:
    """Harvested cases whose scope this test can reconstruct exactly."""
    return [case for case in REGRESSION_CASES if case["scope"] in _SCOPES]


def test_every_deliberate_disagreement_names_a_real_harvested_case():
    """Guard: an exclusion that matches nothing is an exclusion hiding a later miss.

    If a harvested declaration is edited or dropped, its entry here stops matching and
    silently stops excluding -- which is safe -- but it also leaves a stale reason in
    the file that reads as if it were still load-bearing. This fails instead.
    """
    harvested = {case["text"] for case in REGRESSION_CASES}
    stale = sorted(set(_DELIBERATE_DISAGREEMENTS) - harvested)
    assert not stale, f"these exclusions match no harvested case: {stale}"


def test_the_corpus_loads_and_its_labels_are_inside_the_schema():
    """Offline guard: a label the classifier cannot answer would measure nothing."""
    routes = {"risk", "social", "read", "out_of_character", "planner", ""}
    hazards = {"violence", "theft", "destructive", "none", ""}
    for case in REGRESSION_CASES:
        assert case["route"] in routes, case
        assert case["hazard"] in hazards, case


def test_the_skipped_cases_are_reported_rather_than_silently_dropped():
    """Requirement: a bounded corpus must say what it left out.

    Roughly a third of the harvest used bespoke scopes (an active focus, a second
    target, a specific fight roster). Reconstructing each exactly is worth doing, and
    until it is done the count belongs somewhere a reader will see it rather than
    buried in a filter.
    """
    runnable, total = len(_runnable()), len(REGRESSION_CASES)
    skipped = Counter(
        case["scope"] for case in REGRESSION_CASES if case["scope"] not in _SCOPES
    )
    assert runnable > 0
    print(
        f"\nharvested corpus: {runnable}/{total} runnable; "
        f"{total - runnable} skipped for bespoke scopes:"
    )
    for scope, count in skipped.most_common():
        print(f"  {count:3}  {scope[:96]}")


@pytest.mark.live
async def test_the_classifier_never_drops_a_hazard_the_lexicon_caught(tmp_path: Path):
    """The safety direction, enforced exactly.

    Each of these is a declaration some audit round established owes a confirmation.
    The classifier may disagree about which hazard it is -- the categories were never
    the point of those rounds -- but answering ``none`` retires a consent ask an audit
    put there.

    Two exclusions, both about what a ``risk`` label meant in the source test rather
    than about tolerating misses.

    The ``fight`` scope is excluded entirely. In an open fight a ``risk`` route never
    reached a confirmation: ``verify_plan`` calls ``combat_sanctions_violence``
    immediately after, and attacking a recorded combatant is the fight's own mechanic,
    so the ask was bypassed. Those labels therefore say "the lexical trigger fired",
    not "the table was asked", and treating them as owed confirmations would demand the
    classifier flag every in-fight attack so a later rule could unflag it. Out-of-fight
    scopes carry no such bypass, so a ``risk`` label there did mean an ask.

    ``_DELIBERATE_DISAGREEMENTS`` names the individual labels reviewed and rejected.
    """
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    engine._client = object()  # noqa: SLF001 - started-engine seam; only the endpoint is used

    dropped: list[str] = []
    for case in _runnable():
        if case["hazard"] in ("", "none") and case["route"] != "risk":
            continue
        if case["scope"] == "fight":
            continue
        if case["text"] in _DELIBERATE_DISAGREEMENTS:
            continue
        scope, combat = _SCOPES[case["scope"]]
        verdict = await engine.classify_intent(case["text"], scope=scope, combat=combat)
        if verdict is None or verdict.hazard == "none":
            got = "FAULT" if verdict is None else "none"
            dropped.append(f"{case['text']!r} -> {got}  [{case['note'][:70]}]")

    assert not dropped, (
        f"{len(dropped)} declaration(s) an audit pinned as hazardous came back with no "
        "hazard, which retires a consent ask:\n  " + "\n  ".join(dropped)
    )


@pytest.mark.live
async def test_the_classifier_broadly_agrees_with_the_harvested_labels(tmp_path: Path):
    """Breadth: an agreement floor over every positively-labeled harvested case."""
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    engine._client = object()  # noqa: SLF001 - started-engine seam; only the endpoint is used

    agreed = compared = 0
    disagreements: list[str] = []
    for case in _runnable():
        if not case["route"] and not case["hazard"]:
            continue
        scope, combat = _SCOPES[case["scope"]]
        verdict = await engine.classify_intent(case["text"], scope=scope, combat=combat)
        if verdict is None:
            disagreements.append(f"{case['text']!r} -> FAULT")
            compared += 1
            continue
        policy = policy_from(verdict, scope=scope)
        for field, expected, got in (
            ("route", case["route"], policy.route),
            ("hazard", case["hazard"], policy.risk_category),
        ):
            if not expected:
                continue
            compared += 1
            if expected == got:
                agreed += 1
            else:
                disagreements.append(
                    f"{case['text']!r} {field}={got} want {expected}  "
                    f"[{case['note'][:64]}]"
                )

    rate = agreed / compared if compared else 0.0
    report = "\n  ".join(disagreements[:40])
    assert rate >= _AGREEMENT_FLOOR, (
        f"agreement with the harvested labels fell to {agreed}/{compared} "
        f"({rate:.0%}), below the {_AGREEMENT_FLOOR:.0%} floor. "
        f"first disagreements:\n  {report}"
    )
    print(f"\nharvested agreement: {agreed}/{compared} ({rate:.0%})")
