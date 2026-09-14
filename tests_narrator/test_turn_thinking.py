"""The channel agent's thinking level: configuration, request shape, and the terminal command.

Thinking is a per-request endpoint feature (Gemma 4's ``enable_thinking`` template
switch plus vLLM's ``thinking_token_budget`` ceiling), so what this file pins is the
plumbing between a player typing ``/thinking low`` and the exact keys that ride on the
next channel-agent request -- and, just as important, the keys that must *not* ride on
any other lane's request, and the bytes that must not change when the level is off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.channels.base import ThinkingLevelControl  # noqa: E402
from narrator.channels.terminal import TerminalAdapter, _format_status_line  # noqa: E402
from narrator.config import (  # noqa: E402
    DEFAULT_THINKING_BUDGETS,
    THINKING_LEVELS,
    NarratorConfig,
    load_config,
    turn_thinking_from_env,
)
from narrator.engine import NarratorEngine  # noqa: E402
from narrator.model import NarratorOpenAIModel  # noqa: E402
from narrator.prompt import THINKING_SECTION  # noqa: E402
from narrator.service import NarratorService  # noqa: E402

# --- configuration ---------------------------------------------------------------


def test_the_default_level_is_low_and_every_budgeted_level_has_a_positive_budget(tmp_path):
    config = NarratorConfig(campaign_root=tmp_path)
    assert THINKING_LEVELS == ("off", "low", "medium", "high")
    assert config.turn_thinking_level == "low"
    assert config.turn_thinking_budget() == DEFAULT_THINKING_BUDGETS["low"] == 250
    assert config.turn_thinking_budget("off") == 0
    assert [config.turn_thinking_budget(level) for level in ("low", "medium", "high")] == [250, 1000, 2500]


def test_load_config_reads_the_level_and_budgets_from_the_environment(tmp_path, monkeypatch):
    for name in ("BSH_TURN_THINKING", "BSH_TURN_THINKING_BUDGET_LOW",
                 "BSH_TURN_THINKING_BUDGET_MEDIUM", "BSH_TURN_THINKING_BUDGET_HIGH"):
        monkeypatch.delenv(name, raising=False)
    assert load_config(campaign_root=tmp_path).turn_thinking_level == "low"

    monkeypatch.setenv("BSH_TURN_THINKING", "Medium")
    monkeypatch.setenv("BSH_TURN_THINKING_BUDGET_MEDIUM", "750")
    config = load_config(campaign_root=tmp_path)
    assert config.turn_thinking_level == "medium"
    assert config.turn_thinking_budget() == 750
    assert config.turn_thinking_budget("low") == 250  # untouched levels keep their defaults


@pytest.mark.parametrize(
    "environ, message",
    [
        ({"BSH_TURN_THINKING": "max"}, "BSH_TURN_THINKING must be one of"),
        ({"BSH_TURN_THINKING_BUDGET_LOW": "0"}, "BSH_TURN_THINKING_BUDGET_LOW must be a positive integer"),
        ({"BSH_TURN_THINKING_BUDGET_HIGH": "lots"}, "BSH_TURN_THINKING_BUDGET_HIGH must be a positive integer"),
    ],
)
def test_a_malformed_environment_names_the_variable(environ, message):
    with pytest.raises(ValueError, match=message):
        turn_thinking_from_env(environ)


def test_the_dataclass_refuses_an_unknown_level_or_an_unserviceable_budget(tmp_path):
    with pytest.raises(ValueError, match="must be one of"):
        NarratorConfig(campaign_root=tmp_path, turn_thinking_level="extreme")
    with pytest.raises(ValueError, match="key on exactly"):
        NarratorConfig(campaign_root=tmp_path, turn_thinking_budgets={"low": 400})
    with pytest.raises(ValueError, match="positive integer"):
        NarratorConfig(
            campaign_root=tmp_path, turn_thinking_budgets={"low": -1, "medium": 1000, "high": 2500}
        )


# --- the request -----------------------------------------------------------------


def _model(**kwargs) -> NarratorOpenAIModel:
    return NarratorOpenAIModel(
        client_args={"base_url": "http://localhost:8000/v1", "api_key": "not-required"},
        model_id="google/gemma-4-26B-A4B-it",
        params={"temperature": 0.4, "max_tokens": 100},
        **kwargs,
    )


_MESSAGES = [{"role": "user", "content": [{"text": "I leap across the chasm."}]}]


def test_request_overrides_are_read_per_request_not_at_construction():
    level = {"value": {}}
    model = _model(request_overrides=lambda: level["value"])

    first = model.format_request(_MESSAGES, None, "gm")
    assert "extra_body" not in first

    level["value"] = {"extra_body": {"thinking_token_budget": 400}, "max_tokens": 1900}
    second = model.format_request(_MESSAGES, None, "gm")
    assert second["extra_body"] == {"thinking_token_budget": 400}
    assert first["max_tokens"] == 100 and second["max_tokens"] == 1900  # an override wins over params
    # Everything else the request carried is untouched by the override.
    assert {k: v for k, v in second.items() if k not in ("extra_body", "max_tokens")} == {
        k: v for k, v in first.items() if k != "max_tokens"
    }


def test_a_model_without_overrides_sends_the_bytes_it_always_sent():
    assert _model().format_request(_MESSAGES, None, "gm") == _model(
        request_overrides=lambda: {}
    ).format_request(_MESSAGES, None, "gm")


# --- the engine ------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


def _engine(tmp_path, **config_kwargs) -> NarratorEngine:
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path, **config_kwargs))
    engine._system_prompt = "SOUL"
    return engine


def test_off_leaves_the_turn_request_and_prompt_byte_identical(tmp_path):
    engine = _engine(tmp_path, turn_thinking_level="off")
    assert engine._turn_request_overrides() == {}
    assert engine._turn_system_prompt() == "SOUL"
    # The invariant ``_thinking_section_applies`` exists to protect: a session
    # *configured* off sends the bytes every recorded measurement used, and the
    # suppression flag -- which such a session can never actually set, since
    # ``_invoke_channel_agent`` re-raises at off -- does not change that.
    engine._thinking_suppressed = True
    assert engine._turn_request_overrides() == {}
    assert engine._turn_system_prompt() == "SOUL"


@pytest.mark.parametrize("level, budget", [("low", 250), ("medium", 1000), ("high", 2500)])
def test_each_level_sends_the_template_switch_its_budget_and_a_bounded_answer(tmp_path, level, budget):
    engine = _engine(tmp_path, turn_thinking_level=level)
    assert engine._turn_request_overrides() == {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": budget,
        },
        "max_tokens": budget + 1500,
    }
    assert engine._turn_system_prompt() == "SOUL" + THINKING_SECTION


def test_the_answer_bound_never_exceeds_the_turn_ceiling(tmp_path):
    engine = _engine(tmp_path, turn_thinking_level="high", max_tokens=3000)
    assert engine._turn_request_overrides()["max_tokens"] == 3000
    with pytest.raises(ValueError, match="turn_thinking_answer_tokens"):
        NarratorConfig(campaign_root=tmp_path, turn_thinking_answer_tokens=0)


def test_setting_the_level_repoints_every_cached_agent_and_rejects_an_unknown_name(tmp_path):
    engine = _engine(tmp_path)  # default: low
    engine._agents["terminal"] = _FakeAgent(engine._turn_system_prompt())
    engine._agents["second"] = _FakeAgent(engine._turn_system_prompt())
    assert engine._agents["terminal"].system_prompt.endswith(THINKING_SECTION)

    assert engine.set_turn_thinking_level(" OFF ") == "off"
    assert engine.turn_thinking_level == "off"
    assert engine._turn_request_overrides() == {}
    assert all(agent.system_prompt == "SOUL" for agent in engine._agents.values())

    with pytest.raises(ValueError, match="must be one of"):
        engine.set_turn_thinking_level("extreme")
    assert engine.turn_thinking_level == "off"  # a refused name changes nothing

    assert engine.set_turn_thinking_level("high") == "high"
    assert engine._turn_request_overrides()["extra_body"]["thinking_token_budget"] == 2500
    assert all(agent.system_prompt == "SOUL" + THINKING_SECTION for agent in engine._agents.values())


def test_the_request_log_tags_only_the_turn_origin_with_the_level(tmp_path):
    engine = _engine(tmp_path, turn_thinking_level="medium")
    engine._record_request("turn", {"total_chars": 1})
    engine._record_request("classify", {"total_chars": 1})
    assert [row["thinking"] for row in engine._request_log] == ["medium", "off"]


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

    def bind_thinking_level(self, control):
        self.control = control


class _Engine:
    def __init__(self) -> None:
        self.turn_thinking_level = "low"
        self.applied = []

    def start(self):
        return None

    def stop(self):
        return None

    async def flush_pending_sweep(self):
        """This fake never dispatches a background sweep, so nothing to flush;
        exists because ``NarratorService.run()``'s shutdown always awaits it."""
        return None

    def set_turn_thinking_level(self, level):
        self.turn_thinking_level = level
        self.applied.append(level)
        return level

    async def run_turn(self, turn, decision_resolutions=()):  # pragma: no cover
        raise AssertionError("no turn is yielded here")


async def test_the_service_hands_the_adapter_callables_not_the_engine(tmp_path):
    adapter, engine = _Adapter(), _Engine()
    await NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, engine).run()

    control = adapter.control
    assert isinstance(control, ThinkingLevelControl)
    assert control.levels == THINKING_LEVELS
    assert control.read() == "low"
    assert control.write("high") == "high"
    assert engine.applied == ["high"]
    assert control.read() == "high"


async def test_an_engine_without_the_setter_binds_nothing(tmp_path):
    class _Bare(_Engine):
        set_turn_thinking_level = None

    adapter = _Adapter()
    await NarratorService(NarratorConfig(campaign_root=tmp_path), adapter, _Bare()).run()
    assert adapter.control is None


# --- the terminal command ---------------------------------------------------------


class _UI:
    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.rendered = []
        self.status = []

    async def read_line(self, prefix, speaker=""):
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)

    async def render(self, prefix, text):
        self.rendered.append(text)

    async def set_status(self, left, right=""):
        self.status.append((left, right))

    async def close(self):
        return None


def _control(state: dict) -> ThinkingLevelControl:
    def write(level: str) -> str:
        if level not in THINKING_LEVELS:
            raise ValueError(level)
        state["level"] = level
        return level

    return ThinkingLevelControl(levels=THINKING_LEVELS, read=lambda: state["level"], write=write)


async def _drain(adapter):
    return [turn async for turn in adapter.turns()]


async def test_thinking_shows_sets_and_refuses_without_ever_becoming_a_turn():
    state = {"level": "low"}
    ui = _UI(["/thinking", "/thinking high", "/thinking extreme", "/thinking off", "I attack"])
    adapter = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    adapter.bind_thinking_level(_control(state))

    turns = await _drain(adapter)

    assert [turn.mention.text for turn in turns] == ["I attack"]
    assert ui.rendered[0] == "Thinking: low. Levels: off, low, medium, high. Type /thinking followed by a level to change it.\n"
    assert ui.rendered[1] == "Thinking set to high.\n"
    assert ui.rendered[2] == "Unknown thinking level: extreme. Levels: off, low, medium, high.\n"
    assert ui.rendered[3] == "Thinking set to off.\n"
    assert state["level"] == "off"
    # The level is not a status-line fragment: nothing redraws when it changes.
    assert ui.status == []


async def test_an_unbound_terminal_says_so_rather_than_pretending():
    ui = _UI(["/thinking medium"])
    adapter = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    assert await _drain(adapter) == []
    assert ui.rendered == ["The thinking level cannot be changed on this channel.\n"]


async def test_the_legacy_colon_spelling_is_redirected_not_narrated():
    ui = _UI([":thinking low"])
    adapter = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    assert await _drain(adapter) == []
    assert ui.rendered == ["Commands now begin with /. Type /help.\n"]


async def test_help_lists_the_command():
    ui = _UI(["/help"])
    adapter = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    await _drain(adapter)
    assert ui.rendered == ["Commands: /help, /character, /thinking, /language, /quit\n"]


async def test_the_status_line_never_carries_the_level():
    ui = _UI([])
    adapter = TerminalAdapter(author="Rill", character_id="rill", ui=ui)
    adapter.bind_thinking_level(_control({"level": "medium"}))
    await adapter.update_status({"characters": {"rill": {"name": "Rill"}}, "players": {}, "day": 2})
    assert ui.status == [("Rill", "Day 2")]
    assert _format_status_line({"characters": {}, "day": 2}, "rill") == ("rill", "Day 2")


# --- the bounded-loop fallback ----------------------------------------------------


class _MaxTokensReachedException(Exception):  # noqa: N818 -- named to match Strands' own class
    """Named like Strands' class so ``_is_max_tokens_fault`` classifies it by name."""


_MaxTokensReachedException.__name__ = "MaxTokensReachedException"


class _LoopingAgent:
    """Raises the cap once, leaving the partial exchange behind like Strands does."""

    def __init__(self, failures: int = 1) -> None:
        self.messages = [{"role": "user", "content": [{"text": "earlier"}]},
                         {"role": "assistant", "content": [{"text": "Earlier reply."}]}]
        self.system_prompt = ""
        self.failures = failures
        self.calls = []

    async def invoke_async(self, prompt):
        self.calls.append((prompt, self.system_prompt))
        self.messages.append({"role": "user", "content": [{"text": prompt}]})
        if self.failures:
            self.failures -= 1
            self.messages.append({"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "cut"}}}, {"text": "*Wait*, " * 50}]})
            raise _MaxTokensReachedException("Model stopped generating due to maximum token limit.")
        self.messages.append({"role": "assistant", "content": [{"text": "You leap."}]})
        return "You leap."


async def test_a_capped_thinking_attempt_with_no_tool_run_is_retried_once_without_thinking(tmp_path):
    engine = _engine(tmp_path)  # low
    agent = _LoopingAgent()
    agent.system_prompt = engine._turn_system_prompt()
    engine._digest_log.append({})  # this is turn 1

    result = await engine._invoke_channel_agent(agent, "PROMPT")

    assert result == "You leap."
    assert [prompt for prompt, _ in agent.calls] == ["PROMPT", "PROMPT"]


    assert agent.calls[1][1] == "SOUL" + THINKING_SECTION
    assert agent.system_prompt == "SOUL" + THINKING_SECTION
    assert engine.turn_thinking_level == "low" and engine._thinking_suppressed is False
    # The partial exchange is gone; only the successful one follows the earlier history.
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "user", "assistant"]
    assert agent.messages[-1]["content"] == [{"text": "You leap."}]
    assert engine._thinking_fallbacks == [1]


async def test_the_retry_runs_at_off_in_the_request_and_keeps_the_thinking_section(tmp_path):
    """Exactly one of the two consequences of suppression survives.

    The request goes out at ``off``: no thought budget, no template switch, the ceiling
    back to the turn's own. The system prompt does not, because "this table configured
    thinking off" and "this engine is recovering one overrun thought" were the same
    value before ``_thinking_section_applies`` split them, and only the first is a
    reason to drop the section that says which tool to call.
    """
    engine = _engine(tmp_path)
    seen = []

    class _Agent(_LoopingAgent):
        async def invoke_async(self, prompt):
            seen.append((engine._turn_request_overrides(), engine._turn_system_prompt()))
            return await super().invoke_async(prompt)

    await engine._invoke_channel_agent(_Agent(), "PROMPT")
    assert "extra_body" in seen[0][0] and seen[0][1].endswith(THINKING_SECTION)
    assert seen[1] == ({}, "SOUL" + THINKING_SECTION)
    # And the level the request log records for the retry is still the real one: off.
    assert engine._effective_thinking_level() == "low"
    engine._thinking_suppressed = True
    assert engine._effective_thinking_level() == "off"
    assert engine._thinking_section_applies() is True


async def test_no_retry_after_a_tool_ran_or_at_off_or_for_another_fault(tmp_path):
    engine = _engine(tmp_path)
    engine._calls_this_turn = 1  # a tool already committed something this turn
    with pytest.raises(Exception, match="maximum token limit"):
        await engine._invoke_channel_agent(_LoopingAgent(), "PROMPT")
    assert engine._thinking_fallbacks == []

    off = _engine(tmp_path, turn_thinking_level="off")
    with pytest.raises(Exception, match="maximum token limit"):
        await off._invoke_channel_agent(_LoopingAgent(), "PROMPT")

    class _Broken(_LoopingAgent):
        async def invoke_async(self, prompt):
            raise RuntimeError("transport")

    with pytest.raises(RuntimeError):
        await engine._invoke_channel_agent(_Broken(), "PROMPT")


async def test_a_second_cap_propagates_and_restores_the_level(tmp_path):
    engine = _engine(tmp_path)
    agent = _LoopingAgent(failures=2)
    with pytest.raises(Exception, match="maximum token limit"):
        await engine._invoke_channel_agent(agent, "PROMPT")
    assert len(agent.calls) == 2
    assert engine._thinking_suppressed is False and engine._turn_request_overrides() != {}
