"""This file needs no live model and no Strands agent: ``_StubAgent`` stands in for
one, in the exact mold ``tests/test_narrator_units.py`` already uses to pin the
rest of ``run_turn`` (``_StubAgent``, ``_seed_ledger``, ``_engine_with_stub``),
reproduced locally rather than imported across the ``tests/``/``tests_narrator/``
split -- the two trees import nothing from each other anywhere in this
repository.
"""

from __future__ import annotations

from engine_fixtures import _engine_with_stub, _seed_ledger

from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.engine import (
    REPETITION_NUDGE_PROMPT,
    REPETITION_SIMILARITY_THRESHOLD,
    narration_similarity,
    normalize_for_similarity,
)


class _StubResult:
    """The shape ``Agent.invoke_async`` returns: an object carrying ``.message``."""

    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _StubAgent:
    """Stands in for a Strands agent. Replies are consumed in order, one per call.

    The last reply repeats for any call past the end of the list, so a test never
    has to predict exactly how many times the engine will call in -- it only has
    to supply as many distinct replies as it cares to distinguish.
    """

    def __init__(self, replies) -> None:
        self._replies: list[str] = [replies] if isinstance(replies, str) else list(replies)
        if not self._replies:
            raise ValueError("_StubAgent needs at least one reply")
        self.calls: list[str] = []
        self.messages: list[dict] = []
        self.tool_names: list[str] = []

    async def invoke_async(self, text: str):
        self.calls.append(text)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": reply}]})
        return _StubResult(reply)


def _turn(channel_id: str, text: str) -> InboundTurn:
    return InboundTurn(channel_id=channel_id, mention=ChannelMessage("Rill", text))


# -- the normalization and similarity metric ---------------------------------


def test_normalization_folds_case_and_punctuation_not_order():
    """Formatting tolerance only, the same convention as the lenient recall channel."""
    assert normalize_for_similarity("Salt–Cellar!") == "salt cellar"
    assert normalize_for_similarity("  The Stair,\nagain.  ") == "the stair again"
    assert normalize_for_similarity("A—B c") == normalize_for_similarity(
        normalize_for_similarity("A—B c")
    )


def test_similarity_scores_an_exact_repeat_at_one():
    text = "The stairwell continues upward into the gloom, torch guttering."
    assert narration_similarity(text, text) == 1.0


def test_similarity_scores_a_near_verbatim_repeat_above_threshold():
    """One changed word out of many still reads as the same paragraph."""
    a = (
        "The stairwell continues upward, worn stone steps spiraling into the "
        "gloom. Your torch gutters against the damp air. Nothing changes about "
        "the ascent -- just more steps, more shadow, the echo of your own feet."
    )
    b = (
        "The stairwell continues upward, worn stone steps spiraling into the "
        "gloom. Your torch flickers against the damp air. Nothing changes about "
        "the ascent -- just more steps, more shadow, the echo of your own feet."
    )
    assert narration_similarity(a, b) >= REPETITION_SIMILARITY_THRESHOLD


def test_similarity_scores_unrelated_roll_announcements_low():
    """Adversarial case: two turns share the roll-line template but say nothing alike.

    The mandatory ``<Name> rolls <ATTR>: <n> vs <n>, <outcome>.`` announcement line
    (skills/bsh-gm/SKILL.md, "The one invariant") is boilerplate every tested turn
    repeats verbatim. The detector must not fire on that alone.
    """
    a = (
        "Rill rolls DEX: 9 vs 14, success. She catches the ledge before her boot "
        "slips, hauling herself onto the platform above. The tower bell finally "
        "comes into view, tarnished bronze catching the last daylight."
    )
    b = (
        "Ossa rolls WIS: 11 vs 9, success. She hears the faint scrape of claws on "
        "stone somewhere below and waves the party onward. The passage narrows "
        "ahead, forcing everyone to go single file."
    )
    assert narration_similarity(a, b) < REPETITION_SIMILARITY_THRESHOLD


def test_similarity_scores_a_legitimate_partial_repeat_low():
    """Adversarial case: real forward progress that happens to share an opening clause."""
    a = (
        "The stairwell continues upward, worn stone steps spiraling into the "
        "gloom. Your torch gutters against the damp air."
    )
    b = (
        "The stairwell continues upward, but here the steps end at a narrow "
        "landing. A door of black iron stands ajar, cold air spilling through "
        "the gap, and a draft carries the sound of running water from beyond it."
    )
    assert narration_similarity(a, b) < REPETITION_SIMILARITY_THRESHOLD


# -- run_turn wiring -----------------------------------------------------------


async def test_a_first_turn_on_a_channel_never_triggers_the_detector(tmp_path):
    """Nothing to compare against yet, so the candidate delivers on the first call."""
    _seed_ledger(tmp_path, [])
    stub = _StubAgent("The bell is silent.")
    engine = _engine_with_stub(tmp_path, stub)

    outcome = await engine.run_turn(_turn("c1", "look around"))

    assert outcome.narration == "The bell is silent."
    assert len(stub.calls) == 1
    assert engine._repetition_log[-1] == {
        "triggered": False,
        "retried": False,
        "similarity": 0.0,
        "resolved": None,
        "leaks_scrubbed": 0,
    }


async def test_a_genuinely_different_second_turn_never_retries(tmp_path):
    """The common case: different narration, no nudge, one model call."""
    _seed_ledger(tmp_path, [])
    engine = _engine_with_stub(tmp_path, _StubAgent("You reach the landing."))
    await engine.run_turn(_turn("c1", "climb the stairs"))

    stub2 = _StubAgent("A raven watches you from the rafters.")
    engine._agent_for = lambda channel_id: stub2  # noqa: SLF001 - swap the channel agent

    outcome = await engine.run_turn(_turn("c1", "look up"))

    assert outcome.narration == "A raven watches you from the rafters."
    assert len(stub2.calls) == 1
    assert engine._repetition_log[-1]["triggered"] is False


async def test_a_near_identical_repeat_triggers_exactly_one_retry(tmp_path):
    """Test a near identical repeat triggers exactly one retry.
    """
    _seed_ledger(tmp_path, [])
    first = (
        "The stairwell continues upward, worn stone steps spiraling into the "
        "gloom. Your torch gutters against the damp air."
    )
    near_repeat = (
        "The stairwell continues upward, worn stone steps spiraling into the "
        "gloom. Your torch flickers against the damp air."
    )
    advanced = "The steps end at a landing. A black iron door stands ajar ahead."

    engine = _engine_with_stub(tmp_path, _StubAgent(first))
    delivered_first = await engine.run_turn(_turn("c1", "climb the stairs again"))
    assert delivered_first.narration == first

    stub2 = _StubAgent([near_repeat, advanced])
    engine._agent_for = lambda channel_id: stub2  # noqa: SLF001

    outcome = await engine.run_turn(_turn("c1", "climb the stairs again"))

    assert len(stub2.calls) == 2, "exactly one retry, never zero and never more than one"
    assert stub2.calls[1] == REPETITION_NUDGE_PROMPT
    assert outcome.narration == advanced
    assert outcome.narration != delivered_first.narration
    record = engine._repetition_log[-1]
    assert record["triggered"] is True
    assert record["retried"] is True
    assert record["resolved"] is True
    assert record["similarity"] >= REPETITION_SIMILARITY_THRESHOLD


async def test_the_retry_is_bounded_to_one_even_when_the_model_repeats_again(tmp_path):
    """No infinite loop: a model that repeats twice still delivers, on the second try."""
    _seed_ledger(tmp_path, [])
    first = "The bell tolls once, and the square falls quiet."
    still_repeats = "The bell tolls once more, and the square falls quiet again."

    engine = _engine_with_stub(tmp_path, _StubAgent(first))
    await engine.run_turn(_turn("c1", "wait and listen"))

    stub2 = _StubAgent([still_repeats, still_repeats])
    engine._agent_for = lambda channel_id: stub2  # noqa: SLF001

    outcome = await engine.run_turn(_turn("c1", "wait and listen"))

    # Two calls total: the original candidate plus exactly one nudge retry. A
    # detector that kept retrying against a model that keeps repeating would call
    # a third time here; this asserts it does not.
    assert len(stub2.calls) == 2
    assert outcome.narration == still_repeats
    assert outcome.withheld is False, "an exhausted retry still delivers, never silence"
    record = engine._repetition_log[-1]
    assert record["triggered"] is True
    assert record["retried"] is True
    assert record["resolved"] is False


async def test_a_withheld_turn_is_never_the_comparison_point(tmp_path):
    """A turn nobody saw must not stand in for "the prior delivered turn"."""
    _seed_ledger(tmp_path, [])
    first = "The party reaches the shrine gate, torches raised against the coming dark."
    engine = _engine_with_stub(tmp_path, _StubAgent(first))
    delivered = await engine.run_turn(_turn("c1", "approach the gate"))
    assert delivered.withheld is False

    # Turn 2 withholds: an unresolved fiction-debt entry the stub settle path
    # cannot close, so its narration never reaches the table.
    _seed_ledger(tmp_path, [{"seq": 1, "tool": "attribute_test"}])
    unratified = "The party reaches the shrine gate, torches raised, still holding steady."

    async def failing_settle(narration):
        return ("failed", 2, "settler stub broke twice")

    stub2 = _StubAgent(unratified)
    engine._agent_for = lambda channel_id: stub2  # noqa: SLF001
    engine._settle = failing_settle  # noqa: SLF001 - test seam
    withheld_outcome = await engine.run_turn(_turn("c1", "approach the gate"))
    assert withheld_outcome.withheld is True

    # Turn 3 clears the ledger and repeats the ORIGINAL (turn 1) narration almost
    # verbatim. It must still be compared against turn 1 -- the last turn a
    # player actually saw -- and therefore still trigger, proving the withheld
    # turn 2 never became the comparison point.
    _seed_ledger(tmp_path, [])
    near_repeat_of_first = "The party reaches the shrine gate, torches raised against the fading dark."
    advanced = "The gate stands ajar; a bell rope sways though no one pulled it."
    stub3 = _StubAgent([near_repeat_of_first, advanced])
    engine._agent_for = lambda channel_id: stub3  # noqa: SLF001

    outcome = await engine.run_turn(_turn("c1", "approach the gate"))

    assert len(stub3.calls) == 2
    assert outcome.narration == advanced


async def test_channels_are_compared_independently(tmp_path):
    """The same text delivered fresh on a different channel must not trigger."""
    _seed_ledger(tmp_path, [])
    text = "The tide is out, and the flats stretch grey to the horizon."
    engine = _engine_with_stub(tmp_path, _StubAgent(text))
    await engine.run_turn(_turn("channel-a", "look at the flats"))

    stub_b = _StubAgent(text)
    engine._agent_for = lambda channel_id: stub_b  # noqa: SLF001

    outcome = await engine.run_turn(_turn("channel-b", "look at the flats"))

    assert len(stub_b.calls) == 1
    assert outcome.narration == text
    assert engine._repetition_log[-1]["triggered"] is False
