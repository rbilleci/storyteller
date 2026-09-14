"""Pin the M15 fix: a planted fact that never reaches canon fails the soak gate.

``commit_discipline_gate`` is the pure function this fix adds: it reads
``summarize_commit_discipline``'s own ``lost`` list (the entry channel of
``canon_holds``, which needs one single canon statement to carry every word of the
planted fact) and returns the pass/detail pair ``run_soak`` now feeds into
``report.check(\"commit-discipline\", ...)`` whenever ``require_continuity`` is set and
at least one of ``--order-probe``, ``--zone-probe``, or ``--exchange-probe`` planted a
fact. This module needs no live model: ``commit_discipline_gate`` and
``summarize_commit_discipline`` are both pure, reading only the dicts a run already
produced. It lives beside the other soak-gate suites in ``tests_narrator/``,
exercising ``narrator.soak_instruments``.
"""

from __future__ import annotations

import json
from pathlib import Path

from narrator.soak_instruments import (
    SoakReport,
    commit_discipline_gate,
    summarize_commit_discipline,
)


def _commit_discipline_campaign(tmp_path, *facts: str) -> Path:
    """A campaign root whose record holds each given fact, for the summariser to read.

    Reproduces the fixture ``tests/test_soak_instruments.py`` already uses for
    ``summarize_commit_discipline`` itself, so this file exercises the gate against
    the same production data shape rather than a hand-typed stand-in dict.
    """
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    lines = "\n".join(f"- {fact}" for fact in facts)
    (campaign / "scene.md").write_text(
        f"# Scene\n\n## Visible facts\n\n{lines}\n\n## Clocks\n\n- none\n", encoding="utf-8"
    )
    (campaign / "state.json").write_text(json.dumps({"scene": {"summary": ""}}), encoding="utf-8")
    return tmp_path


def test_commit_discipline_gate_passes_when_every_plant_reached_canon(tmp_path):
    """A run where the record holds every planted fact must gate clean."""
    root = _commit_discipline_campaign(
        tmp_path,
        "The tide-medal rides in the waxed pouch at her belt.",
        "The rope ladder is coiled into the canvas sleeve on her back.",
    )
    report = SoakReport(seed=1, window_size=6, generated_messages=41)
    report.tools = {"per_turn": [[] for _ in range(41)], "counts": {}}
    report.tools["per_turn"][19] = ["scene_commit"]
    report.sweep = {"per_turn": ["none"] * 41, "counts": {}}
    posted = [f"narration for turn {index + 1}" for index in range(41)]

    summary = summarize_commit_discipline(
        root, report, posted, {}, {"zone_c_plant": 20, "zone_a_plant": 38}
    )

    passed, detail = commit_discipline_gate(summary)
    assert passed is True
    assert detail == "2 of 2 planted facts reached canon"


def test_commit_discipline_gate_fails_and_names_the_plant_the_record_never_received(tmp_path):
    """A run where one plant's move never reached the record must fail, and say which.

    Reuses the exact refusal-shaped narration ``tests/test_soak_instruments.py``'s
    ``test_the_commit_discipline_block_reports_mentions_and_never_claims_a_statement``
    already pins for ``summarize_commit_discipline`` -- the narration mentions the
    destination inside a sentence declining the move, which is why the gate must read
    the ``entry`` channel (whole-statement containment) rather than a bare mention.
    """
    root = _commit_discipline_campaign(tmp_path, "The oar-case rides in the oilcloth wrap.")
    report = SoakReport(seed=1, window_size=6, generated_messages=30)
    report.tools = {"per_turn": [[] for _ in range(23)], "counts": {}}
    report.sweep = {"per_turn": ["none"] * 23, "counts": {}}
    posted = [""] * 23
    posted[21] = (
        "To seal the oar-case into the tarred tube, you would first have to unpack "
        "the sledge."
    )

    summary = summarize_commit_discipline(
        root, report, posted, {"order_move_first": 12, "order_move_second": 22}, {}
    )

    passed, detail = commit_discipline_gate(summary)
    assert passed is False
    assert "order-move-second" in detail
    assert "lost" in detail


def test_commit_discipline_gate_reads_only_lost_and_ignores_duplicate_counts(tmp_path):
    """The record below states the same fact twice (a duplicate the frozen dedup
    decision leaves unaddressed) while holding every planted fact. The gate must pass:
    it cannot conflate ``duplicate_pairs``/``duplicate_filings`` with a genuine loss,
    which the assignment for this item forbids reopening.
    """
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "scene.md").write_text(
        "# Scene\n\n## Visible facts\n\n"
        "- The tide-medal rides in the waxed pouch at her belt.\n"
        "- The tide-medal rides in the waxed pouch at her belt, still knotted shut.\n"
        "- The rope ladder is coiled into the canvas sleeve on her back.\n"
        "\n## Clocks\n\n- none\n",
        encoding="utf-8",
    )
    (campaign / "state.json").write_text(json.dumps({"scene": {"summary": ""}}), encoding="utf-8")
    report = SoakReport(seed=1, window_size=6, generated_messages=41)
    report.tools = {"per_turn": [[] for _ in range(41)], "counts": {}}
    report.sweep = {"per_turn": ["none"] * 41, "counts": {}}
    posted = [f"narration for turn {index + 1}" for index in range(41)]

    summary = summarize_commit_discipline(
        tmp_path, report, posted, {}, {"zone_c_plant": 20, "zone_a_plant": 38}
    )
    assert summary["duplicate_pairs"] >= 1  # the fixture is genuinely duplicated

    passed, detail = commit_discipline_gate(summary)
    assert passed is True
    assert "duplicate" not in detail


def test_commit_discipline_gate_is_zero_tolerance_not_a_partial_threshold():
    """One lost plant among several still fails the run; the gate admits no quota.
    """
    summary = {"planted": 7, "held_entry": 6, "lost": ["zone-a"]}
    passed, _detail = commit_discipline_gate(summary)
    assert passed is False


def test_commit_discipline_gate_detail_names_every_lost_plant():
    summary = {"planted": 4, "held_entry": 2, "lost": ["order-move-second", "zone-a"]}
    passed, detail = commit_discipline_gate(summary)
    assert passed is False
    assert "order-move-second" in detail
    assert "zone-a" in detail
    assert detail.startswith("2 of 4 planted facts reached canon")
