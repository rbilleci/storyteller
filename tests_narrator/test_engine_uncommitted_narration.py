"""Pin the uncommitted-narration tail `plan_turn` reads by default.

This file needs no live model and no campaign on disk: `_extend_uncommitted_narration`
and `_recent_narration_for` are pure in-memory operations on `NarratorEngine`'s own
per-channel dict, constructible with `NarratorEngine(config)` alone -- `engine.start()`
is never called here. It needs `strands` (`narrator.engine` imports it transitively),
the same reason sibling `tests_narrator/test_engine_*.py` files live here rather than
under `tests/`.
"""

from __future__ import annotations

from pathlib import Path

from fake_classifier import fake_classify_intent

from narrator.config import NarratorConfig
from narrator.engine import UNCOMMITTED_NARRATION_MAX_CHARS, NarratorEngine


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


def test_a_turn_with_no_scene_change_extends_the_tail(tmp_path: Path):
    engine = _engine(tmp_path)
    engine._extend_uncommitted_narration("market", "An iron pulley is bolted to the eastern wall.", scene_changed=False)
    assert engine._recent_narration_for("market") == "An iron pulley is bolted to the eastern wall."

    engine._extend_uncommitted_narration("market", "The wind picks up.", scene_changed=False)
    assert engine._recent_narration_for("market") == (
        "An iron pulley is bolted to the eastern wall.\n\nThe wind picks up."
    )


def test_a_scene_change_clears_the_tail(tmp_path: Path):
    """A commit -- the model's own scene_commit, a settle commit, or the sweep --
    means the digest already carries everything up to this point, so an older tail
    is stale and would only cost tokens restating what canon already has."""
    engine = _engine(tmp_path)
    engine._extend_uncommitted_narration("market", "An iron pulley is bolted to the eastern wall.", scene_changed=False)
    assert engine._recent_narration_for("market")

    engine._extend_uncommitted_narration("market", "The gallery is committed to canon now.", scene_changed=True)
    assert engine._recent_narration_for("market") == ""


def test_the_tail_is_channel_scoped(tmp_path: Path):
    engine = _engine(tmp_path)
    engine._extend_uncommitted_narration("market", "Market fact.", scene_changed=False)
    engine._extend_uncommitted_narration("crypt", "Crypt fact.", scene_changed=False)
    assert engine._recent_narration_for("market") == "Market fact."
    assert engine._recent_narration_for("crypt") == "Crypt fact."
    assert engine._recent_narration_for("unseen-channel") == ""


def test_the_tail_trims_oldest_first_once_over_the_character_bound(tmp_path: Path):
    engine = _engine(tmp_path)
    # The third entry alone is at the bound; combined with the first two, the
    # buffer's total crosses it and trimming must drop the oldest entries first.
    third = "z" * UNCOMMITTED_NARRATION_MAX_CHARS
    engine._extend_uncommitted_narration("market", "first turn, oldest", scene_changed=False)
    engine._extend_uncommitted_narration("market", "second turn, middle", scene_changed=False)
    engine._extend_uncommitted_narration("market", third, scene_changed=False)

    tail = engine._recent_narration_for("market")
    assert "first turn, oldest" not in tail  # trimmed: oldest goes first
    assert third in tail  # the most recent turn always survives
    assert len(tail) <= UNCOMMITTED_NARRATION_MAX_CHARS + len("\n\n")


def test_a_single_turn_longer_than_the_bound_still_survives_whole(tmp_path: Path):
    """Trimming never drops the only entry, however long: a single long turn is
    kept whole rather than silently reduced to nothing."""
    engine = _engine(tmp_path)
    long_narration = "x" * (UNCOMMITTED_NARRATION_MAX_CHARS * 2)
    engine._extend_uncommitted_narration("market", long_narration, scene_changed=False)
    assert engine._recent_narration_for("market") == long_narration


def test_plan_turn_defaults_to_the_tracked_tail_but_an_explicit_value_overrides(tmp_path: Path, monkeypatch):
    """``plan_turn`` itself resolves the default; a caller passing an explicit
    string -- including ``""`` -- overrides that resolution rather than being
    silently replaced by it."""
    import asyncio

    from narrator.channels.base import ChannelMessage, InboundTurn

    engine = _engine(tmp_path)
    engine._extend_uncommitted_narration("probe", "An iron pulley is bolted to the eastern wall.", scene_changed=False)

    seen: list[str] = []

    async def fake_plan_once(prompt, *, repair=False):
        seen.append(prompt)
        return {"plan": {"kind": "proceed"}}

    monkeypatch.setattr(engine, "_plan_once", fake_plan_once)

    turn = InboundTurn("probe", ChannelMessage("Rill", 'I secure a line around the support beam.'))
    asyncio.run(engine.plan_turn(turn, eligible_characters=(("rill", "Rill"),)))
    assert "An iron pulley is bolted to the eastern wall." in seen[0]

    seen.clear()
    asyncio.run(engine.plan_turn(turn, eligible_characters=(("rill", "Rill"),), recent_narration=""))
    assert "An iron pulley is bolted to the eastern wall." not in seen[0]
