"""Terminal-channel and disposable-launcher tests without a model endpoint."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import stat
import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fake_classifier import classification_for

from narrator.channels import terminal_ui
from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.channels.diagnostics import DiagnosticTranscriptRecorder
from narrator.channels.terminal import TerminalAdapter, _format_status_line
from narrator.channels.terminal_ui import PromptToolkitTerminalUI
from narrator.config import NarratorConfig
from narrator.delivery import TurnOutcome
from narrator.service import NarratorService

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeTerminalUI:
    def __init__(self, inputs: list[str | BaseException]) -> None:
        self.inputs = inputs
        self.prompts: list[str] = []
        self.rendered: list[tuple[str, str]] = []
        self.close_calls = 0
        self.statuses: list[tuple[str, str]] = []

    async def read_line(self, prompt_prefix: str, speaker: str = "") -> str:
        self.prompts.append(prompt_prefix)
        value = self.inputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def render(self, prefix: str, text: str) -> None:
        self.rendered.append((prefix, text))

    async def set_status(self, left: str, right: str = "") -> None:
        self.statuses.append((left, right))

    async def close(self) -> None:
        self.close_calls += 1


async def test_terminal_adapter_handles_local_input_and_posts_delivered_text(tmp_path: Path):
    ui = _FakeTerminalUI(["", "/help", "/unknown", "I follow the bell", "/quit"])
    adapter = TerminalAdapter(ui=ui)

    turns = [turn async for turn in adapter.turns()]
    await adapter.post("terminal", "The bell answers from the drowned house.")

    # The default adapter is unpinned and unlinked, so the author label is the
    # neutral "Player" until a campaign link resolves a character.
    assert turns == [
        InboundTurn(
            channel_id="terminal",
            mention=ChannelMessage("Player", "I follow the bell"),
        )
    ]
    assert ui.prompts == ["❯ "] * 5
    assert ui.rendered == [
        ("", "Commands: /help, /character, /thinking, /language, /quit\n"),
        ("", "Unknown command: /unknown. Type /help.\n"),
        ("GM> ", "The bell answers from the drowned house."),
    ]

    target = tmp_path / "terminal.jsonl"
    recorder = DiagnosticTranscriptRecorder(target)
    recorded_ui = _FakeTerminalUI(
        ["/help", "/unknown", "I follow the bell", "/quit"]
    )
    recorded_adapter = TerminalAdapter(recorder=recorder, ui=recorded_ui)
    recorded_turns = [turn async for turn in recorded_adapter.turns()]
    await recorded_adapter.post("terminal", "The bell answers from the drowned house.")
    await recorded_adapter.close()
    result = recorder.close(complete=True)

    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    events = [record["event"] for record in records]

    assert recorded_turns == turns
    assert result.completed is True
    assert result.partial_target.exists() is False
    assert events == [
        "terminal_output",
        "terminal_input",
        "local_command",
        "terminal_output",
        "terminal_output",
        "terminal_input",
        "local_command",
        "terminal_output",
        "terminal_output",
        "terminal_input",
        "terminal_output",
        "terminal_input",
        "local_command",
        "terminal_termination",
        "terminal_output",
    ]
    assert records[1]["text"] == "/help"
    assert records[2]["response"] == "Commands: /help, /character, /thinking, /language, /quit\n"
    assert records[6]["response"] == "Unknown command: /unknown. Type /help.\n"
    assert records[11]["text"] == "/quit"
    assert records[12]["command"] == "/quit"
    assert records[13]["state"] == "local_quit"
    assert records[-1]["kind"] == "delivered_message"
    assert records[-1]["text"] == "GM> The bell answers from the drowned house.\n"


def _sheet_snapshot() -> dict:
    """A snapshot shaped like ``narrator.status.snapshot`` with two linked sheets."""
    return {
        "scene_title": "The eel market", "location_id": "", "day": 1,
        "combat_active": False, "combat_round": None, "combat_active_actor": None,
        "characters": {
            "kara": {
                "name": "Kara", "status": "ok", "hp": 11, "hp_max": 12,
                "doom_die": "d4", "conditions": ["bleeding"],
                "sheet": {
                    "origin": "barbarian", "level": 3, "stories": 1,
                    "backgrounds": ["berserker", "hunter"],
                    "attributes": {
                        "STR": 15, "DEX": 12, "CON": 11, "INT": 9, "WIS": 10, "CHA": 8,
                    },
                    "doom_max": "d6", "weapon_damage": "d8", "unarmed_damage": "d4",
                    "armour": "light", "shield": True,
                    "weapons": ["claymore (two-handed)"],
                    "equipment": ["rope", "flint"],
                    "languages": ["Thyrenian", "Estuary Cant"],
                    "coins": 25,
                    "resources": [{"name": "Rations", "die": "d6"}],
                    "scars": ["split brow"], "gifts": ["second-wind"],
                    "spells": [], "powers": [], "doses": {"healing_balm": 2},
                    "runic_weapon": "", "notes": "Berserker: rage adds a d6.",
                },
            },
            "ossa": {
                "name": "Ossa", "status": "ok", "hp": 9, "hp_max": 9,
                "doom_die": "d6", "conditions": [],
                "sheet": {"origin": "civilised", "level": 1, "coins": 50},
            },
        },
        "players": {"terminal-player": {"character_id": "kara", "display_name": "Kara"}},
    }


async def test_the_character_command_renders_only_this_terminals_own_sheet():
    """``/character`` prints the linked character's sheet from the cached snapshot --
    identity, advancement, attributes, vitals, belongings, and the sections a sheet
    only carries when they exist -- and never another character's data."""
    ui = _FakeTerminalUI(["/character", "/quit"])
    adapter = TerminalAdapter(ui=ui)
    await adapter.update_status(_sheet_snapshot())

    assert [turn async for turn in adapter.turns()] == []
    [(prefix, sheet)] = ui.rendered
    assert prefix == ""
    assert sheet.splitlines() == [
        "# Kara — Barbarian  Level 3",
        "**Backgrounds:** Berserker, Hunter",
        "**Stories:** 1/3",
        "",
        "STR 15  DEX 12  CON 11  INT 9  WIS 10  CHA 8",
        "**HP** 11/12  **Doom** d4 (max d6)  **Armour** light (1) + shield",
        "",
        "**Weapons:** claymore (two-handed) — damage d8, unarmed d4",
        "**Equipment:** rope, flint",
        "**Coins:** 25",
        "**Languages:** Thyrenian, Estuary Cant",
        "**Resources:** Rations d6",
        "**Doses:** Healing Balm ×2",
        "",
        "**Gifts:** Second Wind",
        "**Conditions:** ⚠ bleeding",
        "**Scars:** split brow",
        "",
        "**Notes:**",
        "Berserker: rage adds a d6.",
    ]
    # Nothing of the other linked character leaks onto this terminal.
    assert "Ossa" not in sheet
    assert "50" not in sheet


async def test_the_character_command_before_any_link_states_that_no_sheet_exists():
    """Session zero: no snapshot or no link yet answers with a notice, not a blank."""
    ui = _FakeTerminalUI(["/character", "/quit"])
    adapter = TerminalAdapter(ui=ui)

    assert [turn async for turn in adapter.turns()] == []
    [(_, response)] = ui.rendered
    assert response.startswith("No character sheet yet")


async def test_a_legacy_colon_command_draws_a_redirect_not_a_narrator_turn():
    """The command marker moved from ``:`` to ``/``. A player reaching for the old
    spelling meant a command, and handing ":quit" to the narrator as story text would
    be the worst available reading -- so every formerly valid spelling redirects,
    while colon-leading story text still reaches the narrator untouched."""
    ui = _FakeTerminalUI(
        [":quit", ":help", ":character", ":revise take the bridge", ": a scribble", "/quit"]
    )
    adapter = TerminalAdapter(ui=ui)

    turns = [turn async for turn in adapter.turns()]

    assert [text for _, text in ui.rendered] == ["Commands now begin with /. Type /help.\n"] * 4
    assert [turn.mention.text for turn in turns] == [": a scribble"]


async def test_terminal_adapter_ends_at_end_of_file(tmp_path: Path):
    ui = _FakeTerminalUI([EOFError()])
    adapter = TerminalAdapter(ui=ui)

    assert [turn async for turn in adapter.turns()] == []
    await adapter.close()
    await adapter.close()
    assert ui.prompts == ["❯ "]
    assert ui.close_calls == 2
    target = tmp_path / "terminal.jsonl"
    recorder = DiagnosticTranscriptRecorder(target)
    recorder.record("session_start", channel="terminal")

    result = recorder.close(complete=False)

    assert result.completed is False
    assert target.exists() is False
    assert result.partial_target.exists() is True
    assert [json.loads(line)["event"] for line in result.partial_target.read_text().splitlines()] == [
        "session_start"
    ]

    collision_target = tmp_path / "collision.jsonl"
    collision_recorder = DiagnosticTranscriptRecorder(collision_target)
    collision_recorder.record("session_start", channel="terminal")
    collision_target.write_text("operator transcript\n", encoding="utf-8")

    collision_result = collision_recorder.close(complete=True)

    assert collision_result.completed is False
    assert collision_result.failure.startswith("FileExistsError:")
    assert collision_target.read_text(encoding="utf-8") == "operator transcript\n"
    assert collision_result.partial_target.exists() is True


async def test_terminal_adapter_handles_control_c_and_propagates_idle_cancellation():
    interrupted_ui = _FakeTerminalUI([KeyboardInterrupt()])
    interrupted = TerminalAdapter(ui=interrupted_ui)

    assert [turn async for turn in interrupted.turns()] == []
    assert interrupted._termination_state == "keyboard_interrupt"

    class _BlockingUI(_FakeTerminalUI):
        def __init__(self) -> None:
            super().__init__([])
            self.started = asyncio.Event()

        async def read_line(self, prompt_prefix: str, speaker: str = "") -> str:
            self.prompts.append(prompt_prefix)
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    blocking_ui = _BlockingUI()
    adapter = TerminalAdapter(ui=blocking_ui)
    iterator = adapter.turns()
    receiving = asyncio.create_task(anext(iterator))
    await blocking_ui.started.wait()
    receiving.cancel()

    with pytest.raises(asyncio.CancelledError):
        await receiving
    await adapter.close()
    assert blocking_ui.close_calls == 1


def test_the_status_line_reads_only_this_terminals_own_bound_character():
    snapshot = {
        "scene_title": "The Ashen Bell",
        "location_id": "ashenport",
        "day": 2,
        "combat_active": False,
        "combat_round": None,
        "combat_active_actor": None,
        "characters": {
            "rill": {
                "name": "Rill", "status": "ok", "hp": 7, "hp_max": 10,
                "doom_die": "d6", "conditions": [],
            },
            "ossa": {
                "name": "Ossa", "status": "helpless", "hp": 0, "hp_max": 8,
                "doom_die": "d4", "conditions": ["cursed"],
            },
        },
    }

    assert _format_status_line(snapshot, "rill") == (
        "Rill  HP 7/10  Doom d6", "The Ashen Bell  Day 2"
    )
    # A character absent from the snapshot -- an unreadable sheet -- still names itself
    # by id and still states scene/day, rather than raising or rendering nothing.
    assert _format_status_line(snapshot, "unknown") == ("unknown", "The Ashen Bell  Day 2")


def test_the_status_line_states_life_status_conditions_and_falls_back_to_location_id():
    snapshot = {
        "scene_title": "", "location_id": "ashenport", "day": None,
        "combat_active": False, "combat_round": None, "combat_active_actor": None,
        "characters": {
            "ossa": {
                "name": "Ossa", "status": "helpless", "hp": 0, "hp_max": 8,
                "doom_die": "d4", "conditions": ["cursed", "bleeding"],
            }
        },
    }

    assert _format_status_line(snapshot, "ossa") == (
        "Ossa  HP 0/8 (helpless)  Doom d4  ⚠ cursed, bleeding", "ashenport"
    )


def test_the_status_line_shows_the_round_and_active_actor_during_combat_only():
    snapshot = {
        "scene_title": "The Ashen Bell", "location_id": "ashenport", "day": 2,
        "combat_active": True, "combat_round": 2, "combat_active_actor": "rill",
        "characters": {
            "rill": {
                "name": "Rill", "status": "ok", "hp": 7, "hp_max": 10,
                "doom_die": "d6", "conditions": [],
            }
        },
    }

    assert _format_status_line(snapshot, "rill") == (
        "Rill  HP 7/10  Doom d6", "Combat: Round 2, rill's turn"
    )


async def test_terminal_adapter_forwards_its_own_characters_status_line_to_the_ui():
    ui = _FakeTerminalUI([])
    adapter = TerminalAdapter(ui=ui, character_id="rill")

    await adapter.update_status(
        {
            "scene_title": "The Ashen Bell", "location_id": "", "day": 1,
            "combat_active": False, "combat_round": None, "combat_active_actor": None,
            "characters": {
                "rill": {
                    "name": "Rill", "status": "ok", "hp": 9, "hp_max": 10,
                    "doom_die": "d6", "conditions": [],
                }
            },
        }
    )

    assert ui.statuses == [("Rill  HP 9/10  Doom d6", "The Ashen Bell  Day 1")]


async def test_an_unpinned_adapter_binds_to_the_character_the_campaign_links():
    """The session-zero handshake: the adapter starts unbound, shows a neutral label,
    and binds author, character, and decision filtering to whichever character the
    campaign's own player link names once ``character_create`` writes it. The old
    adapter hardcoded ``rill`` and could never serve a character created live."""
    ui = _FakeTerminalUI([])
    adapter = TerminalAdapter(ui=ui)
    assert adapter.character_id == ""
    assert adapter.author == "Player"

    # Before any link exists (a fresh session-zero campaign): the status line falls
    # back to the same neutral label the prompt echoes, never a blank.
    await adapter.update_status(
        {
            "scene_title": "", "location_id": "", "day": None,
            "combat_active": False, "combat_round": None, "combat_active_actor": None,
            "characters": {}, "players": {},
        }
    )
    assert ui.statuses == [("Player", "")]

    # character_create ran: the link and the sheet now exist, and the next refresh
    # binds this terminal to them without a restart.
    await adapter.update_status(
        {
            "scene_title": "The eel market", "location_id": "", "day": 1,
            "combat_active": False, "combat_round": None, "combat_active_actor": None,
            "characters": {
                "kara": {
                    "name": "Kara", "status": "ok", "hp": 11, "hp_max": 11,
                    "doom_die": "d6", "conditions": [],
                }
            },
            "players": {
                "terminal-player": {"character_id": "kara", "display_name": "Kara"},
            },
        }
    )
    assert adapter.character_id == "kara"
    assert adapter.author == "Kara"
    assert ui.statuses[-1] == ("Kara  HP 11/11  Doom d6", "The eel market  Day 1")

    # A pinned adapter ignores the link: explicit construction stays authoritative.
    pinned = TerminalAdapter(ui=_FakeTerminalUI([]), author="Rill", character_id="rill")
    await pinned.update_status(
        {
            "scene_title": "", "location_id": "", "day": None,
            "combat_active": False, "combat_round": None, "combat_active_actor": None,
            "characters": {},
            "players": {
                "terminal-player": {"character_id": "kara", "display_name": "Kara"},
            },
        }
    )
    assert pinned.character_id == "rill"
    assert pinned.author == "Rill"


async def test_set_thinking_animates_the_status_line_then_restores_it():
    ui = _FakeTerminalUI([])
    adapter = TerminalAdapter(ui=ui, character_id="rill")
    await adapter.update_status(
        {
            "scene_title": "The Ashen Bell", "location_id": "", "day": 1,
            "combat_active": False, "combat_round": None, "combat_active_actor": None,
            "characters": {
                "rill": {
                    "name": "Rill", "status": "ok", "hp": 9, "hp_max": 10,
                    "doom_die": "d6", "conditions": [],
                }
            },
        }
    )
    real_status = ("Rill  HP 9/10  Doom d6", "The Ashen Bell  Day 1")
    assert ui.statuses == [real_status]

    await adapter.set_thinking("terminal", True)
    await asyncio.sleep(0)
    assert len(ui.statuses) >= 2
    assert ui.statuses[-1] != real_status
    # No right half while thinking: the scene it would share the row with cannot
    # have changed yet, since the turn describing it has not resolved.
    assert "Thinking" in ui.statuses[-1][0]
    assert ui.statuses[-1][1] == ""

    # Starting twice is a no-op: the running ticker keeps its own elapsed clock
    # rather than a second one restarting it.
    task = adapter._thinking_task
    await adapter.set_thinking("terminal", True)
    assert adapter._thinking_task is task

    # A snapshot arriving mid-turn must not overwrite the animation the player is
    # currently watching, only update what gets restored once it stops.
    await adapter.update_status(
        {
            "scene_title": "The Ashen Bell", "location_id": "", "day": 1,
            "combat_active": False, "combat_round": None, "combat_active_actor": None,
            "characters": {
                "rill": {
                    "name": "Rill", "status": "ok", "hp": 6, "hp_max": 10,
                    "doom_die": "d6", "conditions": [],
                }
            },
        }
    )
    updated_status = ("Rill  HP 6/10  Doom d6", "The Ashen Bell  Day 1")
    assert "Thinking" in ui.statuses[-1][0]

    await adapter.set_thinking("terminal", False)
    assert ui.statuses[-1] == updated_status
    assert adapter._thinking_task is None

    # Stopping again is a harmless no-op, not a second cancel of a finished task.
    await adapter.set_thinking("terminal", False)
    assert ui.statuses[-1] == updated_status


async def test_post_stops_thinking_and_close_cleans_up_an_active_ticker():
    ui = _FakeTerminalUI([])
    adapter = TerminalAdapter(ui=ui)

    await adapter.set_thinking("terminal", True)
    await asyncio.sleep(0)
    assert adapter._thinking_task is not None

    await adapter.post("terminal", "The bell answers.")
    assert adapter._thinking_task is None
    assert ("GM> ", "The bell answers.") in ui.rendered

    await adapter.set_thinking("terminal", True)
    await asyncio.sleep(0)
    assert adapter._thinking_task is not None
    await adapter.close()
    assert adapter._thinking_task is None
    assert ui.close_calls == 1


async def test_prompt_toolkit_ui_recalls_session_input_without_persistent_history(
    monkeypatch, tmp_path: Path
):
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setenv("HOME", str(tmp_path))
    with create_pipe_input() as pipe:
        ui = PromptToolkitTerminalUI(input=pipe, output=DummyOutput())
        first_input = asyncio.create_task(ui.read_line("Rill> "))
        await asyncio.sleep(0)
        pipe.send_text("I listen at the door\n")
        assert await first_input == "I listen at the door"
        # The fix this pins: one Application keeps running across accepted lines
        # instead of a fresh one per `read_line` call. Before the fix, accepting a
        # line always left the Application `is_done`, which is exactly the condition
        # prompt_toolkit's own bottom-toolbar container filters on -- so the toolbar
        # (and therefore the status line) disappeared the instant a line was accepted.
        first_app_task = ui._app_task
        assert first_app_task is not None
        assert not first_app_task.done()
        assert ui._session.app.is_done is False

        second_input = asyncio.create_task(ui.read_line("Rill> "))
        await asyncio.sleep(0)
        pipe.send_text("I follow the bell\n")
        assert await second_input == "I follow the bell"
        assert ui._app_task is first_app_task
        assert ui._session.app.is_done is False

        recalled_input = asyncio.create_task(ui.read_line("Rill> "))
        await asyncio.sleep(0)
        pipe.send_text("\x1b[A")
        await asyncio.sleep(0)
        pipe.send_text("\x1b[A")
        await asyncio.sleep(0)
        pipe.send_text("\x1b[B")
        await asyncio.sleep(0)
        pipe.send_text("\n")
        assert await recalled_input == "I follow the bell"
        assert isinstance(ui._session.history, InMemoryHistory)
        assert ui._session.app.terminal_size_polling_interval is not None
        await ui.close()
        await ui.close()
    assert list(tmp_path.rglob("*")) == []


async def test_prompt_toolkit_ui_echoes_the_accepted_line_to_real_output(monkeypatch):
    """The regression this pins: keeping one ``Application`` alive across turns (the
    ``4dcc847`` fix for the vanishing status line) also silently stopped the accepted
    line from ever reaching real output. A pseudo-terminal capture of the unfixed code
    showed the exact accept-time redraw (``\\x1b[13D\\x1b[K``, cursor back and erase to
    end of line) erasing the typed text with no prior frame ever having committed it --
    a player's own line vanished the instant they pressed Enter. ``read_line`` must
    route the accepted text through ``print_formatted_text``, the same call ``render``
    already uses for narration, before returning it.

    Also pins the boxed-echo shape: a two-character margin on each side (twice
    ``_BORDER_WIDTH``), the ``speaker`` name and the rest of the line both bold
    (``class:player-prefix class:player-input`` on the name, not just
    ``class:player-prefix`` alone, so the weight reads continuous rather than
    stopping short before the name), a plain "> " separator, then the text --
    padded with an unstyled fill out to the output's own reported width (bold has
    nothing to paint onto blank padding the way a background fill would), and a
    blank, unstyled line on each side setting the box off from whatever comes
    before and after it. ``read_decision_line`` gets no speaker
    (unchanged scope: a decision answer's echo stays nameless), which is its own
    assertion below rather than assumed.
    """
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    captured: list[list[tuple[str, str]]] = []

    def capture(fragments, **kwargs):
        captured.append(list(fragments))

    monkeypatch.setattr(terminal_ui, "print_formatted_text", capture)
    width = DummyOutput().get_size().columns
    margin = ("class:player-border-active", "  ")

    with create_pipe_input() as pipe:
        ui = PromptToolkitTerminalUI(input=pipe, output=DummyOutput())
        first = asyncio.create_task(ui.read_line("❯ ", speaker="Rill"))
        await asyncio.sleep(0)
        pipe.send_text("I attack Rade with my knife\n")
        assert await first == "I attack Rade with my knife"

        assert captured, "the accepted line was never echoed to output"
        text = "I attack Rade with my knife"
        content_width = len("Rill" + "> " + text)
        assert captured[-1] == [
            ("", "\n"),
            margin,
            ("class:player-prefix class:player-input", "Rill"),
            # "> " and the text share one style (class:player-input), so the word-wrap
            # pass's own coalescing merges them into a single run rather than leaving
            # the separator as its own fragment.
            ("class:player-input", "> " + text),
            ("", " " * (width - 4 - content_width)),
            margin,
            ("", "\n"),
            ("", "\n"),
        ]

        # A decision answer must echo too -- it is a player line like any other --
        # but with no speaker: read_decision_line's own scope stays nameless.
        decision = asyncio.create_task(ui.read_decision_line("Decision> "))
        await asyncio.sleep(0)
        pipe.send_text("1\n")
        assert await decision == "1"
        assert captured[-1] == [
            ("", "\n"),
            margin,
            ("class:player-input", "1"),
            ("", " " * (width - 4 - 1)),
            margin,
            ("", "\n"),
            ("", "\n"),
        ]

        await ui.close()


async def test_prompt_toolkit_ui_decision_answers_use_the_same_app_but_skip_history(
    monkeypatch, tmp_path: Path
):
    """A decision answer shares the one persistent Application but never joins recall.
    """
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setenv("HOME", str(tmp_path))
    with create_pipe_input() as pipe:
        ui = PromptToolkitTerminalUI(input=pipe, output=DummyOutput())

        ordinary = asyncio.create_task(ui.read_line("Rill> "))
        await asyncio.sleep(0)
        pipe.send_text("I listen at the door\n")
        assert await ordinary == "I listen at the door"
        app_task = ui._app_task

        decision = asyncio.create_task(ui.read_decision_line("Decision> "))
        await asyncio.sleep(0)
        pipe.send_text("2\n")
        assert await decision == "2"
        # The decision read never spun up a second Application, and it restored the
        # normal history afterward rather than leaving the one-shot swap in place.
        assert ui._app_task is app_task
        assert ui._session.default_buffer.history is ui._session.history

        recalled = asyncio.create_task(ui.read_line("Rill> "))
        # prompt_toolkit's own redraw scheduling (``max_render_postpone_time``) defers a
        # render while the event loop's ready queue is non-empty, so a bare tick isn't
        # always enough for the buffer to finish loading history before a key lands --
        # this needs real wall-clock time, not more ``sleep(0)`` ticks.
        await asyncio.sleep(0.05)
        pipe.send_text("\x1b[A")
        await asyncio.sleep(0.05)
        pipe.send_text("\n")
        # Recall skips straight to the ordinary answer -- the decision's "2" never
        # joined this history, exactly as the one-shot session used to guarantee.
        assert await recalled == "I listen at the door"

        await ui.close()
    assert list(tmp_path.rglob("*")) == []


async def test_prompt_toolkit_ui_maps_control_d_to_end_of_input():
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        ui = PromptToolkitTerminalUI(input=pipe, output=DummyOutput())
        pipe.send_bytes(b"\x04")

        with pytest.raises(EOFError):
            await asyncio.wait_for(ui.read_line("Rill> "), timeout=1)
        await ui.close()


def test_prompt_toolkit_ui_caps_the_live_input_box_at_its_own_content_height():
    """Test prompt toolkit ui caps the live input box at its own content height.
    """
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import BufferControl
    from prompt_toolkit.output import DummyOutput

    from narrator.channels.terminal_ui import _find_buffer_window

    ui = PromptToolkitTerminalUI(output=DummyOutput())
    window = _find_buffer_window(ui._session.layout.container, ui._session.default_buffer)
    assert window is not None
    # No background on this Window any more -- it moved from a shaded box to a
    # top/bottom rule (see the horizontal-rule test below), so nothing here should
    # apply a `bg:`. The height cap this test actually pins is unaffected either way:
    # it comes from `dont_extend_height`, not from whether the window paints anything.
    assert window.style == ""
    assert window.dont_extend_height() is True

    unfixed = Window(BufferControl(buffer=ui._session.default_buffer))
    assert unfixed.dont_extend_height() is False


def test_the_live_input_line_is_framed_by_a_rule_above_and_below_not_shading():
    """The Claude Code idiom this replaced the shaded box with: no background fill on
    the input line itself, instead a full-width horizontal rule immediately above it
    and another immediately below, inserted as real sibling rows in the same
    ``HSplit`` that already holds the input line -- prompt_toolkit has no
    ``top_margin``/``bottom_margin`` the way ``Window`` has ``left_margins``/
    ``right_margins`` for the sides, so a decoration above or below has to be an
    actual sibling container, not an attribute on the existing one.
    """
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.output import DummyOutput

    from narrator.channels.terminal_ui import _RULE_CHAR, _find_split_slot

    ui = PromptToolkitTerminalUI(output=DummyOutput())
    # ``_find_split_slot`` here runs against the *already spliced* layout (the one
    # ``__init__`` built), so the index it returns is the input line's own slot's
    # current position, already shifted past the rule inserted above it -- not the
    # pre-insertion index ``__init__`` itself worked with. The rules sit at one below
    # and one above that current position, not at it and two past it.
    parent, index = _find_split_slot(ui._session.layout.container, ui._session.default_buffer)
    above, middle, below = parent.children[index - 1], parent.children[index], parent.children[index + 1]

    for rule in (above, below):
        assert isinstance(rule, Window)
        assert rule.char == _RULE_CHAR
        assert rule.style == "class:player-border-active"

    # The middle slot is untouched by the splice -- still whatever container
    # PromptSession itself built to hold the input line, now sandwiched between
    # the two new rules rather than replaced by either of them.
    assert above is not middle
    assert below is not middle


async def test_prompt_toolkit_ui_set_status_backs_the_bottom_toolbar_of_both_sessions():
    from prompt_toolkit.output import DummyOutput

    ui = PromptToolkitTerminalUI(output=DummyOutput())

    # ``bottom_toolbar`` is a callable on both the persistent session and any one-shot
    # decision session ``read_decision_line`` builds, so one ``set_status`` call reaches
    # whichever is on screen without the UI needing to track which is active.
    assert ui._session.bottom_toolbar() is None

    await ui.set_status("Rill  HP 7/10  Doom d6")
    assert ui._session.bottom_toolbar() == "Rill  HP 7/10  Doom d6"

    await ui.set_status("")
    assert ui._session.bottom_toolbar() is None

    await ui.close()
    await ui.set_status("after close never applies")
    assert ui._session.bottom_toolbar() is None


async def test_prompt_toolkit_ui_right_justifies_the_status_bars_second_half():
    """The right half is padded out to the terminal's own current width, computed
    fresh each render rather than once in ``set_status`` -- a mid-session resize
    (``DummyOutput``'s own 80 columns stands in for whatever the real terminal
    reports) must not leave it padded for a width that no longer holds.
    """
    from prompt_toolkit.output import DummyOutput

    ui = PromptToolkitTerminalUI(output=DummyOutput())
    width = DummyOutput().get_size().columns

    left, right = "Rill  HP 7/10  Doom d6", "The Ashen Bell  Day 2"
    await ui.set_status(left, right)
    rendered = ui._session.bottom_toolbar()
    assert rendered == left + " " * (width - len(left) - len(right)) + right
    assert len(rendered) == width

    # No right half at all reduces to exactly the left-only path above.
    await ui.set_status(left)
    assert ui._session.bottom_toolbar() == left

    # Two spaces is the floor even when the halves would otherwise collide or
    # overflow, so they never visually merge into one run of text.
    long_left = "x" * (width - 5)
    await ui.set_status(long_left, "y" * 10)
    assert ui._session.bottom_toolbar() == long_left + "  " + "y" * 10

    await ui.close()


async def test_prompt_toolkit_ui_keeps_narration_in_an_unstyled_fragment(monkeypatch):
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    captured = []

    def capture(fragments, **kwargs):
        captured.append((list(fragments), kwargs["end"]))

    monkeypatch.setattr(terminal_ui, "print_formatted_text", capture)
    ui = PromptToolkitTerminalUI(input=DummyInput(), output=DummyOutput())
    narration = "<ansired>still plain</ansired>\nsecond line"

    await ui.render("GM> ", narration)

    # This test guards one property: narration never becomes prompt-toolkit markup --
    # every fragment carrying narration's own text stays unstyled and the angle
    # brackets survive as literal text. It also pins the margin every printed line
    # now gets (two border-styled characters on each side, padded to width, per
    # line -- narration gets no background fill unlike the player's own boxed text).
    assert len(captured) == 1
    fragments, end = captured[0]
    assert end == ""
    width = DummyOutput().get_size().columns
    border = ("class:output-border", "  ")
    line_1_text = "<ansired>still plain</ansired>"
    line_2_text = "second line"
    assert fragments == [
        border,
        ("class:output-prefix", "GM> "),
        ("", line_1_text),
        ("", " " * (width - 4 - len("GM> " + line_1_text))),
        border,
        ("", "\n"),
        border,
        ("", line_2_text),
        ("", " " * (width - 4 - len(line_2_text))),
        border,
        ("", "\n"),
    ]


async def test_render_borders_every_wrapped_row_not_only_the_first(monkeypatch):
    """Caught live: a narration line longer than one terminal row wraps at a point
    the *terminal* chooses, which knows nothing about this module's own margin -- the
    row the wrap started on carried the border, the continuation row the terminal
    created on its own did not, on either side. Word-wrapping inside ``_bordered``
    before the terminal ever sees the text, so no row this module ever prints is wide
    enough to need the terminal's own wrap, is what closes it. This reproduces the
    live report almost verbatim (the Eel Market's own authored description) and
    checks every resulting row, not only the first: each opens and closes with the
    border fragment and reaches exactly the terminal's own width, none of it left to
    the terminal to figure out.
    """
    from prompt_toolkit.formatted_text import fragment_list_width
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    captured = []

    def capture(fragments, **kwargs):
        captured.append(list(fragments))

    monkeypatch.setattr(terminal_ui, "print_formatted_text", capture)
    ui = PromptToolkitTerminalUI(input=DummyInput(), output=DummyOutput())
    width = DummyOutput().get_size().columns

    long_narration = (
        "The eel market is a sprawl of wooden planks on trestles, suspended above "
        "the black, sucking mud of the empty channel. Forty stalls line the "
        "walkways, smelling of salt, smoke, and eel oil."
    )
    await ui.render("GM> ", long_narration)

    assert len(captured) == 1
    fragments = captured[0]
    rows = [[]]
    for style, text in fragments:
        if text == "\n":
            rows.append([])
        else:
            rows[-1].append((style, text))
    if not rows[-1]:
        rows.pop()
    assert len(rows) > 1, "the fixture text must actually wrap for this to prove anything"

    for row in rows:
        assert row[0] == ("class:output-border", "  "), row
        assert row[-1] == ("class:output-border", "  "), row
        assert fragment_list_width(row) == width

    # No character of the original narration was lost to the rewrap, beyond the
    # handful of boundary spaces wrapping is allowed to drop (the same ones a
    # terminal's own native wrap would silently eat too) -- checked with all
    # whitespace collapsed out, since the padding _bordered adds is real space
    # characters too and this test cares about words surviving, not run lengths.
    reconstructed = "".join(text for row in rows for _, text in row)
    assert reconstructed.replace(" ", "") == ("GM> " + long_narration).replace(" ", "")

    await ui.close()


class _IdleAdapter:
    name = "idle"

    def __init__(self) -> None:
        self.receiving = asyncio.Event()
        self.receive_cancelled = asyncio.Event()
        self.closed = False

    async def turns(self):
        self.receiving.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.receive_cancelled.set()
            raise
        if False:  # pragma: no cover - marks this function as an async generator
            yield InboundTurn("idle", ChannelMessage("Rill", "unused"))

    async def post(self, channel_id: str, text: str) -> None:
        raise AssertionError("an idle adapter must not post")

    async def close(self) -> None:
        self.closed = True


class _ActiveAdapter:
    name = "active"

    def __init__(self) -> None:
        self.posted: list[tuple[str, str]] = []
        self.closed = False

    async def turns(self):
        yield InboundTurn("terminal", ChannelMessage("Rill", "I listen"))
        await asyncio.Event().wait()

    async def post(self, channel_id: str, text: str) -> None:
        self.posted.append((channel_id, text))

    async def close(self) -> None:
        self.closed = True


class _Engine:
    def __init__(self, outcome: TurnOutcome) -> None:
        self.outcome = outcome
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    async def classify_intent(
        self, declaration: str, *, scope=None, combat=None, offer_open: bool = False
    ):
        """The offline routing verdict this fake owes the service.

        ``NarratorService`` reaches its classifier through a duck-typed
        ``classify_intent`` and routes no turn without one, so a fake that omits it
        withholds every turn and these tests time out waiting for ``run_turn``. See
        ``tests_narrator/lexical_double.py`` for what the double does and does not
        establish.
        """
        return classification_for(
            declaration, scope=scope, combat=combat, offer_open=offer_open
        )

    async def run_turn(self, turn: InboundTurn) -> TurnOutcome:
        self.started.set()
        await self.release.wait()
        return self.outcome

    async def flush_pending_sweep(self) -> None:
        """This fake never dispatches a background sweep, so nothing to flush;
        exists because ``NarratorService.run()``'s shutdown always awaits it."""

    def stop(self) -> None:
        self.stop_calls += 1


class _ThinkingTrackingAdapter(_ActiveAdapter):
    """An adapter recording every ``set_thinking`` call, without TerminalAdapter's own
    stop-on-post behaviour -- that half is TerminalAdapter's own implementation choice,
    not part of the ``ThinkingIndicatorAdapter`` protocol's contract, and is already
    pinned directly against the real adapter above."""

    def __init__(self) -> None:
        super().__init__()
        self.thinking_calls: list[tuple[str, bool]] = []

    async def set_thinking(self, channel_id: str, thinking: bool) -> None:
        self.thinking_calls.append((channel_id, thinking))


async def test_service_starts_thinking_before_the_engine_call_and_before_stop(
    tmp_path: Path,
):
    """Requirement pinned here rather than only by inspection: ``NarratorService.run``
    must flip an optional ``set_thinking`` indicator on the instant a turn is picked
    up -- before the engine call that can take real wall-clock time -- and never
    require the adapter to implement it at all (a plain adapter with no such method
    must run exactly as before).
    """
    adapter = _ThinkingTrackingAdapter()
    engine = _Engine(TurnOutcome("The reeds bend toward Vey.", ratified=True, withheld=False))
    service = NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine)

    running = asyncio.create_task(service.run())
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    # Set before the engine call resolves, not after -- a player waiting on a reply
    # cannot distinguish decision routing from the engine's own generation time, and
    # this is checked while the engine call is still deliberately blocked on
    # ``engine.release`` to prove the ordering, not just the eventual end state.
    assert ("terminal", True) in adapter.thinking_calls

    engine.release.set()
    service.request_stop()
    await asyncio.wait_for(running, timeout=1)

    # A plain adapter with no ``set_thinking`` (``_IdleAdapter``/``_ActiveAdapter``'s
    # own siblings, exercised throughout this file) never raises reaching this same
    # code path -- proven by every other ``test_service_*`` test in this file passing
    # unchanged after this feature landed, not restated here.


async def test_service_stop_cancels_only_an_idle_adapter_receive(tmp_path: Path):
    adapter = _IdleAdapter()
    engine = _Engine(TurnOutcome("unused", ratified=True, withheld=False))
    service = NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine)

    running = asyncio.create_task(service.run())
    await asyncio.wait_for(adapter.receiving.wait(), timeout=1)
    assert engine.start_calls == 1
    service.request_stop()
    report = await asyncio.wait_for(running, timeout=1)

    assert report.turns == 0
    assert adapter.receive_cancelled.is_set()
    assert adapter.closed is True
    assert engine.start_calls == 1
    assert engine.stop_calls == 1


class _StatusLineIdleAdapter(_IdleAdapter):
    """An idle adapter that also records every status-line snapshot it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.statuses: list[dict] = []

    async def update_status(self, snapshot: dict) -> None:
        self.statuses.append(snapshot)


async def test_the_status_line_is_populated_before_the_first_turn(tmp_path: Path):
    """A channel with a status line shows real values before any input, not a blank one.

    Every other refresh point (``narrator.delivery.refresh_status``) is a reaction to
    something already delivered, so without this, a freshly bootstrapped campaign --
    which already has real starting HP and a seeded scene -- showed nothing until the
    player's first turn came back.
    """
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "state.json").write_text(
        '{"scene": {"title": "The Ashen Bell"}, "fiction_debt": [], "pending_rulings": []}',
        encoding="utf-8",
    )
    (campaign / "characters").mkdir()
    (campaign / "characters" / "rill.json").write_text(
        '{"id": "rill", "name": "Rill", "hp": 10, "hp_max": 10, "doom_die": "d6"}',
        encoding="utf-8",
    )
    adapter = _StatusLineIdleAdapter()
    engine = _Engine(TurnOutcome("unused", ratified=True, withheld=False))
    service = NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine)

    running = asyncio.create_task(service.run())
    await asyncio.wait_for(adapter.receiving.wait(), timeout=1)
    # The refresh happens before the adapter is ever asked for a turn. The nested
    # sheet (the ``/character`` command's data, pinned by ``tests/test_status.py``)
    # rides the same snapshot; the flat vitals contract is asserted exactly.
    [snapshot] = adapter.statuses
    assert snapshot["characters"]["rill"].pop("sheet")["origin"] == ""
    assert adapter.statuses == [{
        "scene_title": "The Ashen Bell", "location_id": "", "day": None,
        "combat_active": False, "combat_round": None, "combat_active_actor": None,
        "characters": {
            "rill": {
                "name": "Rill", "status": "ok", "hp": 10, "hp_max": 10,
                "doom_die": "d6", "conditions": [],
            }
        },
        "players": {},
    }]

    service.request_stop()
    await asyncio.wait_for(running, timeout=1)


async def test_service_finishes_an_active_turn_after_stop_requested(tmp_path: Path):
    adapter = _ActiveAdapter()
    engine = _Engine(TurnOutcome("The reeds bend toward Vey.", ratified=True, withheld=False))
    service = NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine)

    running = asyncio.create_task(service.run())
    await asyncio.wait_for(engine.started.wait(), timeout=1)
    service.request_stop()
    await asyncio.sleep(0)
    assert running.done() is False

    engine.release.set()
    report = await asyncio.wait_for(running, timeout=1)

    assert report.turns == 1
    assert adapter.posted == [("terminal", "The reeds bend toward Vey.")]
    assert adapter.closed is True
    assert engine.stop_calls == 1


async def test_service_stops_the_engine_when_adapter_close_fails(tmp_path: Path):
    class _FailingCloseAdapter(_IdleAdapter):
        async def turns(self):
            if False:  # pragma: no cover - marks this function as an async generator
                yield InboundTurn("idle", ChannelMessage("Rill", "unused"))

        async def close(self) -> None:
            raise RuntimeError("close failed")

    adapter = _FailingCloseAdapter()
    engine = _Engine(TurnOutcome("unused", ratified=True, withheld=False))
    engine.release.set()
    service = NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine)

    with pytest.raises(RuntimeError, match="close failed"):
        await service.run()
    assert engine.stop_calls == 1


def _play_terminal_module():
    specification = importlib.util.spec_from_file_location(
        "play_terminal", REPO_ROOT / "scripts" / "play_terminal.py"
    )
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _narrator_serve_module():
    specification = importlib.util.spec_from_file_location(
        "narrator_serve_for_test", REPO_ROOT / "scripts" / "narrator_serve.py"
    )
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    engine = types.ModuleType("narrator.engine")
    engine.DISCLOSED_SKILLS = ()
    engine.NarratorEngine = object
    previous_engine = sys.modules.get("narrator.engine")
    sys.modules["narrator.engine"] = engine
    try:
        specification.loader.exec_module(module)
    finally:
        if previous_engine is None:
            del sys.modules["narrator.engine"]
        else:
            sys.modules["narrator.engine"] = previous_engine
    return module


def test_terminal_bootstrap_seeds_one_rill_and_one_player_link(tmp_path: Path):
    launcher = _play_terminal_module()
    launcher.bootstrap_campaign(REPO_ROOT, tmp_path, sys.executable)

    game = launcher.GameService(tmp_path)
    character_ids = game.store.character_ids()
    players = game.store.read_players().players

    assert game.store.read_state().scene.title == "The eel market at low water"
    assert character_ids == ["rill"]
    assert game.store.read_character("rill").name == "Rill"
    assert players[0].discord_user_id == "terminal-player"
    assert players[0].character_id == "rill"


def test_terminal_bootstrap_supports_a_premade_for_every_origin(tmp_path: Path):
    """Every supported origin has a jump-in path, each landing on the same seeded,
    audited sheet run to run — including the decadent's Forbidden Knowledge spell
    draws, the origin whose creation exercises the sorcery subsystem."""
    launcher = _play_terminal_module()
    for origin, premade in launcher.PREMADES.items():
        sheets = []
        for attempt in ("a", "b"):
            root = tmp_path / f"{origin}-{attempt}"
            root.mkdir()
            launcher.bootstrap_campaign(REPO_ROOT, root, sys.executable, origin)
            game = launcher.GameService(root)
            [character_id] = game.store.character_ids()
            character = game.store.read_character(character_id)
            players = game.store.read_players().players
            assert character.name == premade["name"]
            assert character.origin == origin
            assert character.backgrounds == [b.lower() for b in premade["backgrounds"]]
            assert character.hp == character.hp_max == character.attributes.CON
            assert character.doom_die == "d6"
            assert character.weapons, f"{origin} premade carries no weapons"
            assert players[0].discord_user_id == "terminal-player"
            assert players[0].character_id == character_id
            assert game.store.read_state().scene.title == "The eel market at low water"
            sheet = character.model_dump()
            # Wall-clock stamps are the one legitimately varying field; everything
            # rolled or derived must reproduce exactly.
            sheet.pop("created_at", None)
            sheet.pop("updated_at", None)
            sheets.append(sheet)
        assert sheets[0] == sheets[1], f"{origin} premade is not reproducible"

    decadent = tmp_path / "decadent-a"
    vessa = launcher.GameService(decadent).store.read_character("vessa")
    assert len(vessa.spells) == 4, "Forbidden Knowledge must draw four starting spells"


def test_terminal_bootstrap_session_zero_starts_with_nothing_durable(tmp_path: Path):
    """The session-zero launch hands the narrator a campaign with no characters, no
    player links, and no seeded scene: every sheet and the first scene must enter
    through the tools during the session itself."""
    launcher = _play_terminal_module()
    launcher.bootstrap_campaign(REPO_ROOT, tmp_path, sys.executable, launcher.SESSION_ZERO)

    game = launcher.GameService(tmp_path)
    assert game.store.character_ids() == []
    assert game.store.read_players().players == []
    assert game.store.read_state().scene.title == ""
    # The campaign is still fully valid and serviceable: the same status call the
    # narrator makes on its first turn succeeds with an empty party.
    status = game.campaign_status()
    assert status["ok"] is True
    assert status["party"] == []


def test_launcher_builds_terminal_child_and_cleans_its_campaign(tmp_path: Path, monkeypatch):
    launcher = _play_terminal_module()
    observed: dict[str, object] = {}

    class _Child:
        def wait(self) -> int:
            return 17

        def send_signal(self, number: int) -> None:
            raise AssertionError(f"unexpected signal {number}")

    def fake_popen(command, *, env, stderr):
        observed["command"] = command
        observed["environment"] = env
        observed["campaign_root"] = Path(command[3])
        observed["stderr"] = stderr
        assert Path(command[3]).is_dir()
        return _Child()

    project_python = sys.executable
    messages = io.StringIO()
    status = launcher.launch(
        REPO_ROOT,
        {"NARRATOR_PYTHON": "/test/narrator-python", "HOME": str(tmp_path)},
        project_python,
        fake_popen,
        status_stream=messages,
    )

    command = observed["command"]
    environment = observed["environment"]
    campaign_root = observed["campaign_root"]
    assert status == 17
    assert command[:6] == [
        "/test/narrator-python",
        str(REPO_ROOT / "scripts" / "narrator_serve.py"),
        "--root",
        str(campaign_root),
        "--channel",
        "terminal",
    ]
    assert command[-2] == "--diagnostic-transcript"
    automatic_target = Path(command[-1])
    assert automatic_target.parent == (
        tmp_path / ".local" / "state" / "storyteller" / "diagnostics"
    )
    assert stat.S_IMODE(automatic_target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(automatic_target.parent.parent.stat().st_mode) == 0o700
    status_log_target = automatic_target.with_suffix(".stderr.log")
    assert messages.getvalue() == (
        f"Diagnostic transcript: {automatic_target}\n"
        f"Narrator status log: {status_log_target}\n"
    )
    assert environment["BSH_SERVER_PYTHON"] == project_python
    assert campaign_root.exists() is False


    assert observed["stderr"] is not None
    assert observed["stderr"].name == str(status_log_target)
    assert stat.S_IMODE(status_log_target.stat().st_mode) == 0o600
    target = tmp_path / "terminal.jsonl"

    command = launcher.serve_command(
        REPO_ROOT,
        tmp_path / "campaign",
        {"NARRATOR_PYTHON": "/test/narrator-python"},
        target,
    )

    assert command[-2:] == ["--diagnostic-transcript", str(target)]
    assert launcher.serve_command(
        REPO_ROOT,
        tmp_path / "campaign",
        {"NARRATOR_PYTHON": "/test/narrator-python"},
        None,
    )[-2:] == ["--channel", "terminal"]

    narrator_serve = _narrator_serve_module()
    campaign_target = tmp_path / "campaign.jsonl"
    with pytest.raises(ValueError, match="outside the campaign root"):
        narrator_serve._resolve_external_diagnostic_target(str(campaign_target), tmp_path)

    external_target = tmp_path.parent / f"{tmp_path.name}-terminal.jsonl"

    async def return_creation_failure_status(arguments, transcript_log, recorder):
        del arguments, transcript_log
        assert recorder is None
        return 37

    def fail_to_create(target):
        del target
        raise OSError("unavailable")

    monkeypatch.setattr(narrator_serve, "_main", return_creation_failure_status)
    monkeypatch.setattr(narrator_serve, "DiagnosticTranscriptRecorder", fail_to_create)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "narrator_serve.py",
            "--channel",
            "terminal",
            "--root",
            str(tmp_path),
            "--diagnostic-transcript",
            str(external_target),
        ],
    )

    assert narrator_serve.main() == 37

    class _FailingCloseRecorder:
        def record(self, event, **fields) -> None:
            del event, fields

        def close(self, *, complete):
            del complete
            raise OSError("finalization unavailable")

    async def return_finalization_failure_status(arguments, transcript_log, recorder):
        del arguments, transcript_log
        assert isinstance(recorder, _FailingCloseRecorder)
        return 41

    monkeypatch.setattr(narrator_serve, "_main", return_finalization_failure_status)
    monkeypatch.setattr(
        narrator_serve,
        "DiagnosticTranscriptRecorder",
        lambda target: _FailingCloseRecorder(),
    )

    assert narrator_serve.main() == 41

    def fail_default_target(repo_root, environment, *args, **kwargs):
        # Stands in for default_diagnostic_target itself, which both the diagnostic
        # transcript path and (through default_status_log_target) the status log path
        # resolve through; it must fail the same way regardless of which keyword
        # arguments a caller adds.
        del repo_root, environment, args, kwargs
        raise OSError("state directory unavailable")

    monkeypatch.setattr(launcher, "default_diagnostic_target", fail_default_target)
    monkeypatch.setattr(launcher, "bootstrap_campaign", lambda *args: None)
    messages = io.StringIO()
    observed.clear()

    assert launcher.launch(
        REPO_ROOT,
        {"NARRATOR_PYTHON": "/test/narrator-python", "HOME": str(tmp_path)},
        project_python,
        fake_popen,
        status_stream=messages,
    ) == 17
    assert observed["command"][-2:] == ["--channel", "terminal"]
    # default_diagnostic_target is patched to fail unconditionally, so both the
    # diagnostic transcript and the status log -- which falls back to deriving its own
    # target from the same function once no transcript was resolved -- fail the same
    # way. The child must still never inherit this process's stderr: the one
    # acceptable fallback is subprocess.DEVNULL.
    assert messages.getvalue() == (
        "warning: diagnostic recorder unavailable: OSError\n"
        "warning: narrator status log unavailable: OSError\n"
    )
    assert observed["stderr"] == subprocess.DEVNULL


def test_launcher_default_target_observes_xdg_fallback_and_unique_names(tmp_path: Path):
    launcher = _play_terminal_module()
    fixed = datetime(2000, 1, 1, 0, 0, 0, 1500, tzinfo=UTC)
    xdg_root = tmp_path / "state"

    first = launcher.default_diagnostic_target(
        REPO_ROOT,
        {"XDG_STATE_HOME": str(xdg_root), "HOME": str(tmp_path / "home")},
        now=fixed,
        process_id=123,
        token="first",
    )
    second = launcher.default_diagnostic_target(
        REPO_ROOT,
        {"XDG_STATE_HOME": str(xdg_root), "HOME": str(tmp_path / "home")},
        now=fixed,
        process_id=123,
        token="second",
    )

    assert first.parent == xdg_root / "storyteller" / "diagnostics"
    assert first != second
    assert first.name == "terminal-20000101T000000.001500Z-123-first.jsonl"

    fallback = launcher.default_diagnostic_target(
        REPO_ROOT,
        {"XDG_STATE_HOME": "relative-state", "HOME": str(tmp_path / "home")},
        now=fixed,
        process_id=123,
        token="fallback",
    )
    assert fallback.parent == (
        tmp_path / "home" / ".local" / "state" / "storyteller" / "diagnostics"
    )

    with pytest.raises(ValueError, match="HOME must be absolute"):
        launcher.default_diagnostic_target(
            REPO_ROOT,
            {"XDG_STATE_HOME": "relative-state", "HOME": "relative-home"},
        )

    with pytest.raises(ValueError, match="outside the repository"):
        launcher.default_diagnostic_target(
            REPO_ROOT,
            {"XDG_STATE_HOME": str(REPO_ROOT / "diagnostics")},
        )


def test_launcher_cli_defaults_overrides_and_opts_out(monkeypatch, tmp_path: Path):
    launcher = _play_terminal_module()
    selected = []
    modes = []

    def fake_launch(*, diagnostic_transcript, mode):
        selected.append(diagnostic_transcript)
        modes.append(mode)
        return 0

    monkeypatch.setattr(launcher, "launch", fake_launch)

    monkeypatch.setattr(sys, "argv", ["play_terminal.py"])
    assert launcher.main() == 0
    assert selected[-1] is launcher._DEFAULT_DIAGNOSTIC
    assert modes[-1] == "barbarian"

    override = tmp_path / "chosen.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        ["play_terminal.py", "--diagnostic-transcript", str(override)],
    )
    assert launcher.main() == 0
    assert selected[-1] == override

    monkeypatch.setattr(sys, "argv", ["play_terminal.py", "--no-diagnostic-transcript"])
    assert launcher.main() == 0
    assert selected[-1] is None

    monkeypatch.setattr(sys, "argv", ["play_terminal.py", "--premade", "decadent"])
    assert launcher.main() == 0
    assert modes[-1] == "decadent"

    monkeypatch.setattr(sys, "argv", ["play_terminal.py", "--session-zero"])
    assert launcher.main() == 0
    assert modes[-1] == launcher.SESSION_ZERO

    # The two modes are one choice: a launch is either a premade jump-in or a
    # session zero, never both.
    monkeypatch.setattr(
        sys, "argv", ["play_terminal.py", "--session-zero", "--premade", "civilised"]
    )
    with pytest.raises(SystemExit):
        launcher.main()


async def test_terminal_channel_rejects_non_tty_before_engine_start(monkeypatch, tmp_path):
    narrator_serve = _narrator_serve_module()

    class _Stream:
        def __init__(self, tty: bool) -> None:
            self.tty = tty

        def isatty(self) -> bool:
            return self.tty

    class _Recorder:
        def __init__(self) -> None:
            self.records = []

        def record(self, event, **fields) -> None:
            self.records.append((event, fields))

    recorder = _Recorder()
    monkeypatch.setattr(narrator_serve.sys, "stdin", _Stream(False))
    monkeypatch.setattr(narrator_serve.sys, "stdout", _Stream(True))
    arguments = types.SimpleNamespace(
        channel="terminal",
        root=str(tmp_path),
        base_url="http://localhost:8000/v1",
        model="model",
        no_commit_nudge=False,
    )

    assert await narrator_serve._main(arguments, None, recorder) == 2
    assert recorder.records == [
        ("session_start", {"channel": "terminal"}),
        ("error", {"category": "terminal_not_tty"}),
        ("session_termination", {"state": "preflight_failed"}),
    ]


def test_recorder_uses_private_modes_and_preserves_partial_on_transport_failure(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "private" / "nested" / "terminal.jsonl"
    recorder = DiagnosticTranscriptRecorder(target)

    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.parent.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(recorder.partial_target.stat().st_mode) == 0o600

    recorder.record("session_start", channel="terminal")
    result = recorder.close(complete=True)

    assert result.completed is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o600

    narrator_serve = _narrator_serve_module()
    failed_target = tmp_path.parent / f"{tmp_path.name}-transport-failure.jsonl"

    async def fail_transport(arguments, transcript_log, diagnostic_recorder):
        del arguments, transcript_log, diagnostic_recorder
        raise RuntimeError("prompt toolkit transport failure")

    monkeypatch.setattr(narrator_serve, "_main", fail_transport)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "narrator_serve.py",
            "--channel",
            "terminal",
            "--root",
            str(tmp_path),
            "--diagnostic-transcript",
            str(failed_target),
        ],
    )

    with pytest.raises(RuntimeError, match="transport failure"):
        narrator_serve.main()
    assert failed_target.exists() is False
    assert failed_target.with_name(f"{failed_target.name}.partial").exists() is True


def test_launcher_preserves_state_and_events_before_deleting_the_campaign(
    tmp_path: Path,
):
    """A disposable campaign deletes its logs at exit; retention copies them out first."""
    launcher = _play_terminal_module()
    observed: dict[str, object] = {}

    class _Child:
        def wait(self) -> int:
            return 0

        def send_signal(self, number: int) -> None:
            raise AssertionError(f"unexpected signal {number}")

    def fake_popen(command, *, env, stderr):
        campaign_root = Path(command[3])
        observed["campaign_root"] = campaign_root
        observed["target"] = Path(command[-1])
        observed["stderr"] = stderr
        # Stand in for the narrator child: write the two evidence files the real
        # server produces during a session, and a status line onto the routed-away
        # stderr handle -- proving it is a real, writable file this test can read back.
        (campaign_root / "campaign" / "logs").mkdir(parents=True, exist_ok=True)
        (campaign_root / "campaign" / "state.json").write_text(
            '{"event_seq": 3}\n', encoding="utf-8"
        )
        (campaign_root / "campaign" / "logs" / "events.jsonl").write_text(
            '{"tool": "combat_attack"}\n', encoding="utf-8"
        )
        stderr.write("a library warning that must never reach the play TTY\n")
        stderr.flush()
        return _Child()

    status_stream = io.StringIO()
    status = launcher.launch(
        REPO_ROOT,
        {"NARRATOR_PYTHON": "/test/narrator-python", "HOME": str(tmp_path)},
        sys.executable,
        fake_popen,
        status_stream=status_stream,
    )

    assert status == 0
    campaign_root = observed["campaign_root"]
    target = observed["target"]
    assert campaign_root.exists() is False  # the disposable campaign is gone
    preserved_state = target.with_suffix(".state.json")
    preserved_events = target.with_suffix(".events.jsonl")
    assert preserved_state.read_text(encoding="utf-8") == '{"event_seq": 3}\n'
    assert preserved_events.read_text(encoding="utf-8") == '{"tool": "combat_attack"}\n'
    status_log_target = target.with_suffix(".stderr.log")
    assert observed["stderr"].name == str(status_log_target)
    assert status_log_target.read_text(encoding="utf-8") == (
        "a library warning that must never reach the play TTY\n"
    )
    assert "a library warning" not in status_stream.getvalue()
    assert stat.S_IMODE(preserved_state.stat().st_mode) == 0o600
    assert stat.S_IMODE(preserved_events.stat().st_mode) == 0o600


def test_preserve_campaign_evidence_skips_absent_sources_without_raising(tmp_path: Path):
    """A campaign that logged nothing must copy nothing and raise nothing."""
    launcher = _play_terminal_module()
    campaign_root = tmp_path / "camp"
    (campaign_root / "campaign").mkdir(parents=True)  # no state.json, no logs
    target = tmp_path / "diagnostics" / "run.jsonl"
    target.parent.mkdir(parents=True)

    launcher.preserve_campaign_evidence(campaign_root, target, io.StringIO())

    assert not target.with_suffix(".state.json").exists()
    assert not target.with_suffix(".events.jsonl").exists()


def test_preserve_campaign_evidence_is_gated_on_a_transcript_target(tmp_path: Path):
    """Disabling the diagnostic transcript disables retention with it."""
    launcher = _play_terminal_module()
    campaign_root = tmp_path / "camp"
    (campaign_root / "campaign" / "logs").mkdir(parents=True)
    (campaign_root / "campaign" / "state.json").write_text("{}", encoding="utf-8")

    # A None target must copy nothing and raise nothing.
    launcher.preserve_campaign_evidence(campaign_root, None, io.StringIO())
    assert list(tmp_path.glob("*.state.json")) == []


def test_markdown_reaches_the_player_rendered_rather_than_as_markers():
    """Regression: emphasis and list markers printed verbatim.

    The renderer emitted the narration as one unstyled fragment, so a player read
    ``**Doom**`` and a leading hyphen instead of bold text and a bullet.
    """
    fragments = terminal_ui.markdown_fragments("The **Doom** die turns.")
    assert ("class:markdown-strong", "Doom") in fragments
    assert "".join(text for _, text in fragments) == "The Doom die turns."

    assert ("class:markdown-emphasis", "soft") in terminal_ui.markdown_fragments("A *soft* sound.")
    assert ("class:markdown-emphasis", "lone") in terminal_ui.markdown_fragments("A _lone_ word.")
    assert ("class:markdown-code", "combat_defend") in terminal_ui.markdown_fragments(
        "Call `combat_defend` now."
    )

    bullets = terminal_ui.markdown_fragments("- first\n- second\n")
    assert "".join(text for _, text in bullets) == "• first\n• second\n"

    heading = terminal_ui.markdown_fragments("# Heading\nbody\n")
    assert ("class:markdown-strong", "Heading") in heading
    assert "".join(text for _, text in heading) == "Heading\nbody\n"


def test_the_renderer_never_swallows_a_character_of_narration():
    """Untrusted narration must survive whatever markup it carries.

    The guarantee is not that every span styles correctly; it is that no visible
    character disappears. A renderer that ate narration would be worse than the defect
    it replaces, because a player cannot see what is missing.
    """
    import random
    import string

    # The alphabet carries every character the renderer's rules read, including the tab
    # and the plus that the bullet and heading rules accept, which an audit found missing.
    random.seed(20260808)
    alphabet = string.ascii_letters + string.digits + " .,;:!?'\"()[]{}<>/\\|@$%^&=~" + "*_`-+#\t\n"
    markers = set("*_`-+#")
    for _ in range(4000):
        source = "".join(random.choice(alphabet) for _ in range(random.randint(0, 40)))
        rendered = "".join(text for _, text in terminal_ui.markdown_fragments(source))
        # Every non-marker, non-whitespace character survives. That is stronger than
        # comparing only alphanumerics, because a rule consuming punctuation would pass
        # the weaker check. Whitespace is excluded because a bullet and a heading marker
        # legitimately consume the space that follows them.
        assert [c for c in source if c not in markers and not c.isspace()] == [
            c for c in rendered if c not in markers and not c.isspace() and c != "\u2022"
        ], source


def test_unmatched_markup_and_identifiers_survive_as_literal_text():
    """Failing toward literal text is what keeps a partial marker readable.

    "Roll under <13" was a recorded leak-filter defect in this repository's history. A
    renderer deleting to end of line on an unbalanced marker would repeat that class of
    harm. An identifier's inner underscores must not open an emphasis span either.
    """
    for source in (
        "An unmatched ** marker stays.",
        "combat_defend_directive stays whole.",
        "2 * 3 * 4 = 24",
        "snake_case_name and another_one",
        "*",
        "_",
        "",
    ):
        rendered = "".join(text for _, text in terminal_ui.markdown_fragments(source))
        assert rendered == source, source


def test_a_bullet_needs_its_space_so_emphasis_opening_a_line_still_reads():
    """``*italic*`` at line start is emphasis, and ``* item`` is a bullet."""
    emphasised = terminal_ui.markdown_fragments("*italic* opens the line")
    assert ("class:markdown-emphasis", "italic") in emphasised
    assert "•" not in "".join(text for _, text in emphasised)

    bullet = terminal_ui.markdown_fragments("* item")
    assert "".join(text for _, text in bullet) == "• item"


def test_plain_narration_renders_exactly_as_before():
    """Narration carrying no markup produces one unstyled fragment, as it always did."""
    source = "The lamp gutters and the water rises against the pilings."
    assert terminal_ui.markdown_fragments(source) == [("", source)]


# --- Autocomplete: SigilCompleter, TerminalAdapter wiring, NarratorService roster ---


def _completions(completer, text: str):
    from prompt_toolkit.document import Document

    document = Document(text, cursor_position=len(text))
    return list(completer.get_completions(document, None))


def test_sigil_completer_offers_commands_by_prefix_only_at_a_slash_token():
    from narrator.channels.terminal_ui import SigilCompleter

    commands = (("retry", "Try again"), ("revise", "Replace the last action"), ("quit", "Leave"))
    completer = SigilCompleter(commands=lambda: commands, mentions=lambda: ())

    matches = _completions(completer, "/re")
    assert [item.text for item in matches] == ["/retry", "/revise"]
    assert matches[0].display_meta_text == "Try again"
    # Not preceded by whitespace or start-of-line: a path or a fraction, not a command.
    assert _completions(completer, "http://re") == []
    # No leading marker at all: ordinary narration text.
    assert _completions(completer, "I retry") == []


def test_sigil_completer_offers_mentions_by_substring_at_an_at_token():
    from narrator.channels.base import MentionCandidate
    from narrator.channels.terminal_ui import SigilCompleter

    mentions = (
        MentionCandidate(sigil="GM", kind="gm"),
        MentionCandidate(sigil="Ossa", kind="pc"),
        MentionCandidate(sigil="Sera Vane", kind="npc"),
    )
    completer = SigilCompleter(commands=lambda: (), mentions=lambda: mentions)

    matches = _completions(completer, "tell @se")
    assert [item.text for item in matches] == ["@Sera Vane"]
    # Accepting inserts exactly the sigil form; no description is attached to a mention.
    assert matches[0].display_meta_text == ""

    assert [item.text for item in _completions(completer, "@")] == ["@GM", "@Ossa", "@Sera Vane"]


def test_sigil_completer_replaces_only_the_partial_token():
    from narrator.channels.terminal_ui import SigilCompleter

    completer = SigilCompleter(commands=lambda: (("retry", ""),), mentions=lambda: ())
    matches = _completions(completer, "please /re")
    assert matches[0].start_position == -3


def test_prompt_toolkit_ui_wires_the_given_completer_into_its_session():
    from prompt_toolkit.output import DummyOutput

    from narrator.channels.terminal_ui import SigilCompleter

    completer = SigilCompleter(commands=lambda: (), mentions=lambda: ())
    ui = PromptToolkitTerminalUI(output=DummyOutput(), completer=completer)
    assert ui._session.completer is completer

    bare = PromptToolkitTerminalUI(output=DummyOutput())
    assert bare._session.completer is None


async def test_prompt_toolkit_ui_reserves_no_extra_rows_until_a_menu_is_actually_open():
    """Regression pin: PromptSession's own ``reserve_space_for_menu`` inflates the
    input line's own window to a fixed minimum the instant ``complete_while_typing``
    is enabled -- for this whole session, not only while a menu is genuinely open --
    which read as the input box permanently padded with blank lines. This renders a
    real screen (piped input, a ``Vt100_Output`` writing into a string buffer) to pin
    both halves of the fix: idle stays exactly one row, and an open menu actually
    renders its candidates rather than reserving blank space nothing draws into.
    """
    import io

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output

    from narrator.channels.terminal_ui import SigilCompleter

    def height(ui) -> int:
        return ui._session.app.renderer._last_screen.height

    def rendered_text(ui) -> str:
        screen = ui._session.app.renderer._last_screen
        lines = []
        for y in range(height(ui)):
            row = screen.data_buffer.get(y, {})
            maxcol = max(row.keys()) if row else 0
            lines.append("".join(row.get(x, type("_", (), {"char": " "})()).char for x in range(maxcol + 1)))
        return "\n".join(lines)

    completer = SigilCompleter(
        commands=lambda: (("retry", "Try again"), ("revise", "Replace")), mentions=lambda: ()
    )
    with create_pipe_input() as pipe:
        output = Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=80))
        ui = PromptToolkitTerminalUI(input=pipe, output=output, completer=completer)
        task = asyncio.ensure_future(ui.read_line("> "))
        await asyncio.sleep(0.05)
        assert height(ui) == 3  # rule, input, rule -- no reserved blank menu space

        pipe.send_text("/re")
        await asyncio.sleep(0.2)
        assert height(ui) == 5
        opened = rendered_text(ui)
        assert "/retry" in opened
        assert "/revise" in opened

        # A narrator reply interleaving here (``ui.render``) is what a real session
        # does between reads and is what settles the renderer's own pending state in
        # this harness -- without it, ``_last_screen`` can lag a frame behind rapid
        # piped-input changes with no real per-keystroke wall-clock gaps between them.
        await ui.render("", "")
        pipe.send_text("\x08\x08\x08")
        await asyncio.sleep(0.2)
        assert height(ui) == 3  # back to no reservation once nothing is completing

        pipe.send_text("\x1b\n")
        await asyncio.wait_for(task, timeout=1)
        await ui.close()


def test_terminal_adapter_offers_recovery_commands_only_while_recovery_is_active():
    ui = _FakeTerminalUI([])
    adapter = TerminalAdapter(ui=ui)

    names = {name for name, _ in adapter._command_specs()}
    assert "retry" not in names
    assert "help" in names and "character" in names

    adapter._decision_recovery_active = True
    names = {name for name, _ in adapter._command_specs()}
    assert {"retry", "revise", "continue", "dismiss"} <= names


def test_terminal_adapter_mention_candidates_delegate_to_the_bound_control():
    from narrator.channels.base import MentionCandidate, MentionDirectoryControl

    ui = _FakeTerminalUI([])
    adapter = TerminalAdapter(ui=ui)
    assert adapter._mention_candidates() == ()

    roster = (MentionCandidate(sigil="Ossa", kind="pc"),)
    adapter.bind_mentions(MentionDirectoryControl(candidates=lambda: roster))
    assert adapter._mention_candidates() == roster


def test_terminal_adapter_mention_candidates_omit_this_terminals_own_player():
    from narrator.channels.base import MentionCandidate, MentionDirectoryControl

    ui = _FakeTerminalUI([])
    adapter = TerminalAdapter(ui=ui, author="Ossa")
    roster = (
        MentionCandidate(sigil="GM", kind="gm"),
        MentionCandidate(sigil="Ossa", kind="pc"),
        MentionCandidate(sigil="Sera Vane", kind="npc"),
    )
    adapter.bind_mentions(MentionDirectoryControl(candidates=lambda: roster))

    candidates = adapter._mention_candidates()
    assert MentionCandidate(sigil="Ossa", kind="pc") not in candidates
    assert MentionCandidate(sigil="GM", kind="gm") in candidates
    assert MentionCandidate(sigil="Sera Vane", kind="npc") in candidates


def test_service_mention_candidates_builds_the_roster_from_campaign_files(tmp_path: Path):
    from narrator.channels.base import MentionCandidate

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "players.yaml").write_text(
        "players:\n"
        '  - discord_user_id: "acct-1"\n'
        '    character_id: "ossa"\n'
        '    display_name: "Ossa"\n',
        encoding="utf-8",
    )
    (campaign / "state.json").write_text(
        json.dumps(
            {
                "npcs": {
                    "sera-vane": {"name": "Sera Vane", "status": "alive"},
                    "orso-pell": {"name": "Orso Pell", "status": "dead"},
                },
                "scene": {"persons": {"rade": {"name": "Rade", "role": "fishmonger"}}},
            }
        ),
        encoding="utf-8",
    )
    adapter = TerminalAdapter(ui=_FakeTerminalUI([]))
    engine = _Engine(TurnOutcome("", ratified=True, withheld=False))
    service = NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine)

    roster = service._mention_candidates()
    assert MentionCandidate(sigil="GM", kind="gm") in roster
    assert MentionCandidate(sigil="Ossa", kind="pc") in roster
    assert MentionCandidate(sigil="Sera Vane", kind="npc") in roster
    assert MentionCandidate(sigil="Rade", kind="person") in roster
    assert not any(candidate.sigil == "Orso Pell" for candidate in roster)


def test_service_mention_candidates_fails_open_on_missing_campaign_files(tmp_path: Path):
    from narrator.channels.base import MentionCandidate

    adapter = TerminalAdapter(ui=_FakeTerminalUI([]))
    engine = _Engine(TurnOutcome("", ratified=True, withheld=False))
    service = NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine)

    assert service._mention_candidates() == (MentionCandidate(sigil="GM", kind="gm"),)


def test_service_start_binds_mentions_to_an_adapter_that_offers_them(tmp_path: Path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    adapter = TerminalAdapter(ui=_FakeTerminalUI([]))
    engine = _Engine(TurnOutcome("", ratified=True, withheld=False))
    service = NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine)

    assert adapter._mention_control is None
    service._bind_mentions()
    assert adapter._mention_control is not None
    assert adapter._mention_candidates() == service._mention_candidates()
