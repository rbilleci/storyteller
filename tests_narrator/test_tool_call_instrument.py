"""The duplicate-call instrument: same-name-same-argument attempts within one turn.

Signatures are process-local and dropped at each turn boundary; only the per-turn count
is retained, so the diagnostics' no-arguments contract holds.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.config import NarratorConfig  # noqa: E402
from narrator.engine import NarratorEngine, duplicate_call_count, tool_call_signature  # noqa: E402


def test_duplicate_call_count_is_repeats_of_the_same_signature():
    a1 = tool_call_signature("dice_roll", {"sides": 20})
    a2 = tool_call_signature("dice_roll", {"sides": 20})
    b = tool_call_signature("dice_roll", {"sides": 6})
    assert a1 == a2 and a1 != b
    assert duplicate_call_count([]) == 0
    assert duplicate_call_count([a1, b]) == 0
    assert duplicate_call_count([a1, a2]) == 1
    assert duplicate_call_count([a1, a2, a2, b]) == 2


def test_the_signature_is_argument_order_independent():
    assert tool_call_signature("x", {"a": 1, "b": 2}) == tool_call_signature(
        "x", {"b": 2, "a": 1}
    )


def test_the_engine_logs_one_duplicate_count_per_turn(tmp_path):
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    assert engine._duplicate_call_log == []
    sig = tool_call_signature("dice_roll", {"sides": 20})
    engine._tool_signatures_this_turn = [sig, sig]
    engine._append_tool_log()
    assert engine._duplicate_call_log == [1]
