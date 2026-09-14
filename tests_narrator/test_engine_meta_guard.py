"""Pin the meta-turn narration guard in ``NarratorEngine.run_turn``.

These are pure functions with no engine dependency (``META_TOOL_NAME_PATTERN``,
``META_PROMISE_PATTERN``, ``_meta_leak_detected``, ``_drop_meta_leak_sentences``),
tested directly, plus ``run_turn``-level wiring tests using the same
``_StubAgent``/``_engine_with_stub`` mold ``tests_narrator/test_engine_repetition.py``
and ``tests/test_narrator_units.py`` both already use.
"""

from __future__ import annotations

from engine_fixtures import _engine_with_stub, _seed_ledger

from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.engine import (
    META_PROMISE_PATTERN,
    META_TOOL_NAME_PATTERN,
    _drop_meta_leak_sentences,
    _meta_leak_detected,
)
from narrator.service import PreparedNarrationTurn

# -- the detection patterns ---------------------------------------------------


def test_the_live_leak_matches_both_patterns():
    """The exact live-session sentence: a backticked tool name and a promise."""
    text = "(I will now call `combat_defend` for Vessa to resolve this.)"
    assert META_TOOL_NAME_PATTERN.search(text)
    assert META_PROMISE_PATTERN.search(text)
    assert _meta_leak_detected(text)


def test_a_bare_tool_name_alone_is_detected():
    assert _meta_leak_detected("Let me invoke npc_create to bring him into the scene.")


def test_a_promise_with_no_tool_name_is_still_detected():
    assert _meta_leak_detected("I will roll the attack for Rade now.")
    assert _meta_leak_detected("I'll resolve this with the dice.")


def test_ordinary_out_of_fiction_prose_is_not_flagged():
    """An answer that explains the game, rather than acting in it, names no engine
    tool by its literal identifier and promises nothing right now."""
    for text in (
        "As the game master, I control the NPCs.",
        "It is currently Rade's turn in combat.",
        "I am asking because it is currently your turn.",
        "The rules say attacks require a roll under your attribute.",
        "I will explain how a roll works.",
    ):
        assert not _meta_leak_detected(text), text


def test_the_tool_name_pattern_excludes_the_meta_turns_own_allowed_reads():
    """``campaign_status``, ``character_sheet``, and the rest of
    ``policy.GM_DISCUSSION_TOOLS`` stay legal to call on a meta turn, so mentioning
    one is not a leak -- only a refused tool's name is."""
    assert not META_TOOL_NAME_PATTERN.search("Let me check campaign_status for you.")


# -- the sentence-drop fallback ------------------------------------------------


def test_drop_sentences_keeps_the_fiction_and_drops_only_the_promise():
    text = (
        "**Rade lashes out with a heavy, salt-crusted knife, aiming for Vessa's "
        "midsection.**\n\n(I will now call `combat_defend` for Vessa to resolve this.)"
    )
    cleaned, dropped = _drop_meta_leak_sentences(text)
    assert cleaned == (
        "**Rade lashes out with a heavy, salt-crusted knife, aiming for Vessa's "
        "midsection.**"
    )
    assert dropped == 1


def test_drop_sentences_drops_only_the_leaking_sentence_of_several():
    text = "As the game master, I control the NPCs. I will now resolve Rade's action."


    cleaned, dropped = _drop_meta_leak_sentences(text)
    assert cleaned == "As the game master, I control the NPCs."
    assert dropped == 1


def test_drop_sentences_is_a_true_no_op_on_entirely_clean_text():
    text = "As the game master, I control the NPCs. It is Rade's turn."
    cleaned, dropped = _drop_meta_leak_sentences(text)
    assert cleaned == text
    assert dropped == 0


def test_drop_sentences_returns_empty_when_every_sentence_leaks():
    cleaned, dropped = _drop_meta_leak_sentences("I will now call combat_attack.")
    assert cleaned == ""
    assert dropped == 1


# -- run_turn wiring -----------------------------------------------------------


class _StubResult:
    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _StubAgent:
    """Replies are consumed in order, one per call; the last repeats past the end."""

    def __init__(self, replies) -> None:
        self._replies: list[str] = [replies] if isinstance(replies, str) else list(replies)
        self.calls: list[str] = []
        self.messages: list[dict] = []

    async def invoke_async(self, text: str):
        self.calls.append(text)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": reply}]})
        return _StubResult(reply)


def _meta_turn(text: str) -> PreparedNarrationTurn:
    return PreparedNarrationTurn(
        source=InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", text)),
        interaction_cue=None,
        turn_framing="gm_discussion",
        meta=True,
    )


async def test_a_clean_meta_reply_never_triggers_the_guard(tmp_path):
    _seed_ledger(tmp_path, [])
    stub = _StubAgent("As the game master, I control the NPCs.")
    engine = _engine_with_stub(tmp_path, stub)

    outcome = await engine.run_turn(_meta_turn("@GM why?"))

    assert outcome.narration == "As the game master, I control the NPCs."
    assert engine._meta_guard_log[-1]["triggered"] is False
    assert len(stub.calls) == 1  # no corrective re-invoke


async def test_a_leaking_reply_is_corrected_by_one_retry(tmp_path):
    _seed_ledger(tmp_path, [])
    stub = _StubAgent(
        [
            "(I will now call `combat_defend` for Vessa to resolve this.)",
            "I am asking because it is currently Rade's turn.",
        ]
    )
    engine = _engine_with_stub(tmp_path, stub)

    outcome = await engine.run_turn(_meta_turn("@GM why are you asking me?"))

    assert outcome.narration == "I am asking because it is currently Rade's turn."
    assert len(stub.calls) == 2
    record = engine._meta_guard_log[-1]
    assert record["triggered"] is True
    assert record["retried"] is True
    assert record["resolved"] is True
    assert record["sentences_dropped"] == 0


async def test_a_retry_that_still_leaks_falls_back_to_dropping_the_bad_sentence(tmp_path):
    _seed_ledger(tmp_path, [])
    stub = _StubAgent(
        [
            "Rade lashes out with his knife. (I will now call `combat_defend`.)",
            "Rade attacks. I will now resolve it with combat_defend.",
        ]
    )
    engine = _engine_with_stub(tmp_path, stub)

    outcome = await engine.run_turn(_meta_turn("@GM why are you asking me?"))

    assert outcome.narration == "Rade attacks."
    assert "combat_defend" not in outcome.narration
    record = engine._meta_guard_log[-1]
    assert record["triggered"] is True
    assert record["resolved"] is False
    assert record["sentences_dropped"] == 1


async def test_a_declared_action_never_reaches_the_meta_guard(tmp_path):
    """The guard is meta-only: a real declaration that happens to say "combat_defend"
    in its own fiction (however unlikely) is not this guard's concern."""
    _seed_ledger(tmp_path, [])
    stub = _StubAgent("The knife catches Rade's shoulder.")
    engine = _engine_with_stub(tmp_path, stub)
    turn = PreparedNarrationTurn(
        source=InboundTurn(channel_id="c1", mention=ChannelMessage("Rill", "I attack rade")),
        interaction_cue=None,
        turn_framing="",
        meta=False,
    )

    await engine.run_turn(turn)

    assert engine._meta_guard_log[-1] == {
        "triggered": False, "retried": False, "resolved": None,
        "sentences_dropped": 0, "leaks_scrubbed": 0,
    }
