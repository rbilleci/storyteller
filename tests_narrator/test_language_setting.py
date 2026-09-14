"""The table's language: the ``/language`` terminal command and the service binding.

The launch language is ``BSH_LANGUAGE`` (defaulting to en-US); what this file pins is
the runtime switch -- the plumbing between a player typing ``/language fr`` and the two
catalogs that must flip together: the config-carried narrator domain every notice
property reads, and the terminal adapter's own domain snapshot. A switch that flips
one without the other posts notices in one language under a status line in another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.channels.base import LanguageControl  # noqa: E402
from narrator.channels.terminal import TerminalAdapter  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402
from narrator.locale import available_languages, load  # noqa: E402
from narrator.service import NarratorService  # noqa: E402

LOCALE_ROOT = REPO_ROOT / "locale"


# --- the service binds the control ------------------------------------------------


class _Adapter:
    """A channel that ends immediately and remembers what it was bound to."""

    name = "bound"

    def __init__(self) -> None:
        self.control = None

    async def turns(self):
        return
        yield  # pragma: no cover - makes this an async generator

    async def post(self, channel_id, text):
        return None

    async def close(self):
        return None

    def bind_language(self, control):
        self.control = control


class _Engine:
    def start(self):
        return None

    def stop(self):
        return None

    async def flush_pending_sweep(self):
        """This fake never dispatches a background sweep, so nothing to flush;
        exists because ``NarratorService.run()``'s shutdown always awaits it."""
        return None

    async def run_turn(self, turn, decision_resolutions=()):  # pragma: no cover
        raise AssertionError("no turn is yielded here")


async def test_the_service_hands_the_adapter_callables_not_the_config(tmp_path):
    adapter = _Adapter()
    config = NarratorConfig(campaign_root=tmp_path)
    await NarratorService(config, adapter, _Engine()).run()

    control = adapter.control
    assert isinstance(control, LanguageControl)
    assert control.languages == available_languages(LOCALE_ROOT)
    assert "fr" in control.languages
    assert control.locale_root == LOCALE_ROOT
    # The system default: English (US), the untouched launch tag.
    assert control.read() == "en-US"
    assert config.fault_notice == load("en", LOCALE_ROOT).notice("fault")

    assert control.write("fr") == "fr"
    assert control.read() == "fr"
    # Every consumer reading through the shared config follows the switch.
    assert config.fault_notice == load("fr", LOCALE_ROOT).notice("fault")

    with pytest.raises(ValueError):
        control.write("xx")
    assert control.read() == "fr", "a refused switch must leave the language standing"


async def test_an_adapter_without_the_binding_binds_nothing(tmp_path):
    class _Bare(_Adapter):
        bind_language = None

    adapter = _Bare()
    await NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, _Engine()).run()
    assert adapter.control is None


# --- the terminal command ---------------------------------------------------------


class _UI:
    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.rendered = []

    async def read_line(self, prefix, speaker=""):
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)

    async def render(self, prefix, text):
        self.rendered.append(text)

    async def set_status(self, left, right=""):
        return None

    async def close(self):
        return None


def _control(state: dict) -> LanguageControl:
    languages = available_languages(LOCALE_ROOT)

    def write(tag: str) -> str:
        if tag not in languages:
            raise ValueError(tag)
        state["language"] = tag
        return tag

    return LanguageControl(
        languages=languages,
        read=lambda: state["language"],
        write=write,
        locale_root=LOCALE_ROOT,
    )


async def _drain(adapter):
    return [turn async for turn in adapter.turns()]


async def test_language_shows_sets_and_refuses_without_ever_becoming_a_turn():
    state = {"language": "en-US"}
    ui = _UI(["/language", "/language xx", "/language fr", "/help", "I attack"])
    adapter = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    adapter.bind_language(_control(state))

    turns = await _drain(adapter)

    assert [turn.mention.text for turn in turns] == ["I attack"]
    languages = ", ".join(available_languages(LOCALE_ROOT))
    assert ui.rendered[0] == (
        f"Language: en-US. Languages: {languages}. "
        "Type /language followed by a language to change it.\n"
    )
    # A refused tag answers in the language still in force, and switches nothing.
    assert ui.rendered[1] == f"Unknown language: xx. Languages: {languages}.\n"
    # The confirmation is the first text the player reads in the new language.
    assert ui.rendered[2] == "Langue réglée sur fr.\n"
    # The adapter reloaded its own domain snapshot: /help now answers in French.
    assert ui.rendered[3] == "Commandes : /help, /character, /thinking, /language, /quit\n"
    assert state["language"] == "fr"


async def test_an_unbound_terminal_says_so_rather_than_pretending():
    ui = _UI(["/language fr"])
    adapter = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    assert await _drain(adapter) == []
    assert ui.rendered == ["The language cannot be changed on this channel.\n"]


async def test_the_legacy_colon_spelling_is_redirected_not_narrated():
    ui = _UI([":language fr"])
    adapter = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    assert await _drain(adapter) == []
    assert ui.rendered == ["Commands now begin with /. Type /help.\n"]


async def test_the_real_control_switches_both_domains_and_back(tmp_path):
    """Through the service-built control, one ``/language`` flips the notices the
    engine posts and the strings the terminal renders, together -- and ``/language en``
    restores the launch presentation."""
    bound = _Adapter()
    config = NarratorConfig(campaign_root=tmp_path)
    await NarratorService(config, bound, _Engine()).run()

    ui = _UI(["/language ja", "/language en"])
    terminal = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    terminal.bind_language(bound.control)
    await _drain(terminal)

    assert ui.rendered[0] == "言語を ja に設定しました。\n"
    assert ui.rendered[1] == "Language set to en.\n"
    assert config.active_language == "en"
    assert config.fault_notice == load("en", LOCALE_ROOT).notice("fault")
