"""The system-prompt instruction naming the table's narration language.

``locale.py`` deliberately never touches model-facing text -- the fourteen engine
notices and friends are the only thing it translates. Until now nothing told the
*model* which language to narrate in either, so it followed whatever language the
player last wrote in rather than the table's configured ``language``. This pins
``narrator.prompt.language_directive`` and its wiring into
``NarratorEngine._turn_system_prompt``: skipped for an English catalog (so an
English, thinking-off session sends the exact bytes every recorded measurement
used), present for any other catalog, and re-applied to a cached channel agent on
every turn so a mid-session ``/language`` switch takes effect without rebuilding the
agent. A live probe found the
system-prompt section alone insufficient (three turns, ``fr-FR``, all delivered in
English though the section was confirmed present in the request), so ``turn_prompt``
also closes every turn with a per-turn reminder naming the same tag -- the mechanism
the effect actually rides on; see ``NarratorEngine._active_language_tag``.
"""

from __future__ import annotations

from engine_fixtures import _seed_ledger

from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.config import NarratorConfig
from narrator.engine import NarratorEngine
from narrator.prompt import language_directive, turn_prompt


def _engine(tmp_path, **config_kwargs) -> NarratorEngine:
    config_kwargs.setdefault("turn_thinking_level", "off")
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, **config_kwargs))
    engine._system_prompt = "SOUL"
    return engine


# -- the unit ------------------------------------------------------------------


def test_an_english_catalog_leaves_the_prompt_byte_identical(tmp_path):
    engine = _engine(tmp_path)  # default: en-US
    assert engine._turn_system_prompt() == "SOUL"


def test_a_non_english_catalog_appends_the_directive_naming_its_own_tag(tmp_path):
    engine = _engine(tmp_path, language="fr")
    assert engine._turn_system_prompt() == "SOUL" + language_directive("fr-FR")


def test_the_thinking_and_language_sections_can_both_be_present(tmp_path):
    from narrator.prompt import THINKING_SECTION

    engine = _engine(tmp_path, language="ja", turn_thinking_level="low")
    prompt = engine._turn_system_prompt()
    assert prompt == "SOUL" + THINKING_SECTION + language_directive("ja-JP")


def test_an_empty_tag_reproduces_the_legacy_closing_instruction():
    assert turn_prompt("I attack").endswith("Resolve this turn.")


def test_a_tag_replaces_the_closing_instruction_with_one_naming_it():
    assert turn_prompt("I attack", language_tag="fr-FR").endswith(
        "Resolve this turn in fr-FR, not English."
    )


# -- run_turn wiring -------------------------------------------------------------


class _StubResult:
    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _StubAgent:
    """Stands in for a Strands agent, in the mold ``test_engine_repetition.py`` uses.

    Replies are consumed in order (last one repeats past the end) so consecutive
    turns read as distinct narration and never trip the repetition guard, which
    would otherwise re-invoke with its own nudge prompt instead of ``turn_prompt``'s.
    """

    def __init__(self, replies) -> None:
        self._replies: list[str] = [replies] if isinstance(replies, str) else list(replies)
        self.system_prompt = ""
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


async def test_a_mid_session_language_switch_repoints_the_cached_agent(tmp_path):
    """The same agent object survives the switch -- only its ``system_prompt`` moves."""
    _seed_ledger(tmp_path, [])
    stub = _StubAgent(["The bell is silent.", "A gull calls overhead.", "The tide turns."])
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, turn_thinking_level="off"))
    engine._system_prompt = "SOUL"
    engine._agent_for = lambda channel_id: stub  # noqa: SLF001 - test seam

    await engine.run_turn(_turn("c1", "look around"))
    assert stub.system_prompt == "SOUL"
    assert stub.calls[-1].endswith("Resolve this turn.")

    engine.config.set_language("fr")
    await engine.run_turn(_turn("c1", "regarde autour"))
    assert stub.system_prompt == "SOUL" + language_directive("fr-FR")
    assert stub.calls[-1].endswith("Resolve this turn in fr-FR, not English.")

    engine.config.set_language("en")
    await engine.run_turn(_turn("c1", "look again"))
    assert stub.system_prompt == "SOUL"
    assert stub.calls[-1].endswith("Resolve this turn.")
