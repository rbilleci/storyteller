"""Pins scripts/probe_world_state.py's contradiction detector.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import probe_world_state as pws  # noqa: E402


def test_detects_the_audited_barrier_opened_claim():
    """The model called a spontaneous STR test, succeeded, and narrated the gate
    breaking open, while ``barrier_state`` stayed \"locked\" in state.json for two
    further turns. The old check never caught this: the party's recorded location
    never moved on this turn, so its \"location moved while locked\" condition was
    never true.
    """
    entry = {
        "turn": 2,
        "location_id": "the-eel-market",
        "barrier_state": "locked",
        "narration": (
            'The gate swings open. You can now reach the road shrine.'
        ),
    }
    reasons = pws._turn_contradicts_state(entry)
    assert reasons
    assert any("barrier itself opened" in reason for reason in reasons)


def test_detects_the_audited_arrival_claim():
    """``location_id`` had not moved, but the narration asserted arrival at the far
    location. The old check never caught this either: by this turn the script-side
    unlock had already flipped ``barrier_state`` away from \"locked\", so the old
    check's \"barrier_state == locked\" half was never true.
    """
    entry = {
        "turn": 4,
        "location_id": "the-eel-market",
        "barrier_state": "open",
        "narration": (
            'You are at the road shrine. The courtyard is empty.'
        ),
    }
    reasons = pws._turn_contradicts_state(entry)
    assert reasons
    assert any("arrival at, or newly-open access to, the far location" in reason for reason in reasons)


def test_a_plain_refusal_is_not_a_contradiction():
    """A flat, correct refusal must never itself be flagged."""
    entry = {
        "turn": 1,
        "location_id": "the-eel-market",
        "barrier_state": "locked",
        "narration": (
            'A locked gate blocks the road shrine. You cannot pass it.'
        ),
    }
    assert pws._turn_contradicts_state(entry) == []


def test_a_negated_refusal_naming_the_far_location_is_not_a_contradiction():
    """Test a negated refusal naming the far location is not a contradiction.
    """
    entry = {
        "turn": 1,
        "location_id": "the-eel-market",
        "barrier_state": "locked",
        "narration": (
            "The iron gate is barred from the customs side. You cannot pass "
            "through to the road shrine while the gate remains closed.\n\nDo you "
            "want to try to force the gate open, search for another way around, "
            "or talk to someone in the market?"
        ),
    }
    assert pws._turn_contradicts_state(entry) == []


def test_a_successful_roll_narrated_as_a_state_change_is_a_contradiction():
    """Test a successful roll narrated as a state change is a contradiction.
    """
    entry = {
        "turn": 3,
        "location_id": "the-eel-market",
        "barrier_state": "locked",
        "narration": (
            "Since you have chosen to force the gate, I will call for a test.\n\n"
            "Rill, you attempt to heave the iron gate open against the heavy "
            "wooden bar on the other side.\n\n**Rill rolls STR: 12 vs 12, "
            "success.**\n\nThe iron groans and the bar shifts with a heavy, "
            "jarring thud against the frame. The gate is no longer fully barred, "
            "though it remains heavy and awkward to move. You can now pass "
            "through to the road shrine.\n\nWould you like to move to the road "
            "shrine now?"
        ),
    }
    assert pws._turn_contradicts_state(entry)


def test_a_failed_force_attempt_is_not_a_contradiction():
    """A roll that fails and leaves the party in place must not itself be flagged."""
    entry = {
        "turn": 3,
        "location_id": "the-eel-market",
        "barrier_state": "locked",
        "narration": (
            "Rill rolls STR: 18 vs 11, failure.\n\nThe iron gate does not budge, "
            "but the heavy clang of your effort against the metal echoes across "
            "the slick walkways. A few traders pause their work, looking toward "
            "you with suspicion.\n\nDo you try a different approach, or do you "
            "look for another way around?"
        ),
    }
    assert pws._turn_contradicts_state(entry) == []


def test_a_true_arrival_after_the_barrier_opened_is_not_a_contradiction():
    """Once the record actually agrees, the identical claim language is not a
    contradiction -- only the mismatch against state.json is."""
    entry = {
        "turn": 5,
        "location_id": "the-road-shrine",
        "barrier_state": "open",
        "narration": (
            "You are at the road shrine. You have passed through the gate and "
            "left the eel market behind."
        ),
    }
    assert pws._turn_contradicts_state(entry) == []


def test_the_self_correcting_turn_is_not_a_contradiction():
    """Test the self correcting turn is not a contradiction.
    """
    entry = {
        "turn": 5,
        "location_id": "the-eel-market",
        "barrier_state": "open",
        "narration": (
            'You remain in the eel market. The road shrine is only a proposed destination. Would you like to move there?'
        ),
    }
    assert pws._turn_contradicts_state(entry) == []


def test_a_withheld_decision_notice_is_not_a_contradiction():
    """A withheld-turn notice carries no claim at all and must not be flagged."""
    entry = {
        "turn": 2,
        "location_id": "the-eel-market",
        "barrier_state": "locked",
        "narration": (
            "No action occurred. This decision segment needs another player "
            "view. Use /continue, /revise, or /dismiss."
        ),
    }
    assert pws._turn_contradicts_state(entry) == []


def test_location_moved_while_locked_is_still_caught():
    """The original structural check -- location moved while the record still
    reads locked -- stays a hard catch even with no narration claim at all,
    unioned with the narration-based checks rather than replaced by them.
    """
    entry = {
        "turn": 2,
        "location_id": "the-road-shrine",
        "barrier_state": "locked",
        "narration": "You are somewhere new.",
    }
    reasons = pws._turn_contradicts_state(entry)
    assert reasons
    assert any("recorded location moved" in reason for reason in reasons)


def test_synthetic_sequence_flags_only_the_two_contradictory_turns():
    """Synthetic sequence flags only the two contradictory turns."""
    turn_log = [
        {
            "turn": 1, "location_id": "the-eel-market", "barrier_state": "locked",
            "narration": (
                'A locked gate blocks the road shrine. You cannot pass it.'
            ),
        },
        {
            "turn": 2, "location_id": "the-eel-market", "barrier_state": "locked",
            "narration": (
                'The gate swings open. You can now reach the road shrine.'
            ),
        },
        {
            "turn": 3, "location_id": "the-eel-market", "barrier_state": "locked",
            "narration": (
                "No action occurred. This decision segment needs another player "
                "view. Use /continue, /revise, or /dismiss."
            ),
        },
        {
            "turn": 4, "location_id": "the-road-shrine", "barrier_state": "open",
            "narration": (
                "No action occurred because the game master could not prepare a "
                "safe decision. Use /retry, /revise, or /dismiss."
            ),
        },
        {
            "turn": 5, "location_id": "the-road-shrine", "barrier_state": "open",
            "narration": (
                "You are at the road shrine. You have passed through the gate "
                "and left the eel market behind."
            ),
        },
        {
            "turn": 6, "location_id": "the-road-shrine", "barrier_state": "open",
            "narration": (
                'You have arrived at the road shrine. A wooden bench stands by the entrance.'
            ),
        },
    ]
    flagged = [entry["turn"] for entry in turn_log if pws._turn_contradicts_state(entry)]
    assert flagged == [2]

    second_attempt_turn_4 = {
        "turn": 4, "location_id": "the-eel-market", "barrier_state": "open",
        "narration": (
            'You are at the road shrine. The courtyard is empty.'
        ),
    }
    assert pws._turn_contradicts_state(second_attempt_turn_4)
