"""Shared header, fixtures and helpers for the split narrator unit-test files.

``test_narrator_units.py`` grew past 5,000 lines across every narrator module;
it is split by module-under-test (core / sweep / canon), and everything the
tests share -- imports, stubs, fixture builders, prompt goldens -- lives here.
``__all__`` deliberately lists underscored helpers so the test files'
``import *`` re-exports them; this module defines no tests and is not collected.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from narrator import delivery, engine, ledger, locale, policy, prompt
from narrator.channels.base import ChannelMessage, InboundTurn
from narrator.config import NarratorConfig
from narrator.server_launch import resolved_server_command, server_env

REPO_ROOT = Path(__file__).resolve().parents[1]


_ROMANCE_NOTICE = locale.load("en", REPO_ROOT / "locale").notice("romance_boundary")


# -- delivery ----------------------------------------------------------------


class _RecordingAdapter:
    name = "recording"

    def __init__(self) -> None:
        self.posted: list[tuple[str, str]] = []

    def turns(self):  # pragma: no cover - unused in these tests
        raise NotImplementedError

    async def post(self, channel_id: str, text: str) -> None:
        self.posted.append((channel_id, text))

    async def close(self) -> None:
        return None


# -- the engine's ratification wiring ----------------------------------------


class _StubResult:
    """The shape ``Agent.invoke_async`` returns: an object carrying ``.message``."""

    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _StubAgent:
    """Stands in for a Strands agent so the barrier tests need no model.

    ``messages`` is the agent's own history list, which a Strands agent carries from
    construction and the engine edits before every turn. The double keeps it so the
    barrier tests exercise the real turn path rather than a shorter one.
    """

    def __init__(self, reply: str = "Narration.") -> None:
        self.reply = reply
        self.calls: list[str] = []
        self.messages: list[dict] = []
        self.tool_names = sorted(policy.PLAYER_FACING_TOOLS)

    async def invoke_async(self, text: str):
        self.calls.append(text)
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": self.reply}]})
        return _StubResult(self.reply)


def _engine_with_stub(campaign_root, stub):
    """An engine whose agent construction is replaced, so no framework is needed."""
    from narrator.engine import NarratorEngine

    engine = NarratorEngine(NarratorConfig(campaign_root=campaign_root))
    engine._agent_for = lambda channel_id: stub  # noqa: SLF001 - test seam
    return engine


def _seed_ledger(campaign_root: Path, entries: list[dict]) -> None:
    campaign = campaign_root / "campaign"
    campaign.mkdir(exist_ok=True)
    (campaign / "state.json").write_text(
        json.dumps({"fiction_debt": entries}), encoding="utf-8"
    )


def _full_debt() -> list[dict]:
    return [
        {
            "seq": 3,
            "tool": "attribute_test",
            "reason": "listen at the crypt door",
            "outcome": "failure",
            "realized": "failure",
            "stakes": {
                "success": "The party hears the keeper moving below.",
                "failure": "The party hears nothing and stays unaware.",
                "hidden": "",
            },
        }
    ]


def _character_entry(character_id: str, name: str, *, coins=None, hp=None, equipment=()) -> dict:
    """One ``_character_resource_values``-shaped entry, for tests that call
    ``guard_ratification`` directly without going through a real character file."""
    return {"id": character_id, "name": name, "coins": coins, "hp": hp, "equipment": tuple(equipment)}


def _npc_entry(npc_id: str, name: str, status: str = "alive") -> dict:
    """One ``_npc_life_values``-shaped entry, for tests that call
    ``guard_ratification`` directly without going through a real state file."""
    return {"id": npc_id, "name": name, "status": status}


def _write_character(tmp_path, character_id: str, **fields) -> None:
    characters = tmp_path / "campaign" / "characters"
    characters.mkdir(parents=True, exist_ok=True)
    payload = {"id": character_id, "name": character_id.title(), "status": "ok",
               "coins": 0, "hp": 1, "hp_max": 1, "equipment": [], "resources": []}
    payload.update(fields)
    (characters / f"{character_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _scene_render(fact_count: int, hidden_count: int = 0, combat: bool = False) -> str:
    """A scene render in the shape ``bsh_mcp.store.render_scene_markdown`` writes.

    The section order is the property under test, so the fixture reproduces it rather
    than importing the renderer: frontmatter, Summary, Visible facts, Exits, Objects,
    Present NPCs, Open hooks, Clocks, Unratified outcomes, and the two game-master
    sections.
    """
    lines = [
        "---",
        "location_id: the-eel-market",
        "session: 1",
        "in_game_minutes: 210",
        "updated_at: 2000-01-01T00:00:00+00:00",
        "---",
        "",
        "# The eel market at slack tide",
        "",
        "## Summary",
        "",
        "The party trades the salt writ for passage while the tide turns.",
        "",
        "## Visible facts",
        "",
    ]
    lines += [
        f"- Fact {index:02d}: the party recorded a durable change at the eel market "
        "before slack tide."
        for index in range(1, fact_count + 1)
    ]
    lines += ["", "## Exits", "", "- the-road-shrine", "- the-drowned-customs-house"]
    lines += ["", "## Objects", "", "- tower-door (locked) an iron slide bolt"]
    lines += ["", "## Present NPCs", "", "- sera-vane"]


    lines += ["", "## Other persons present", "", "- None recorded."]
    if combat:
        # ``CampaignStore._combat_section`` emits this shape, in this position, and
        # emits nothing at all while no fight runs. The default stays off so the
        # 2,398-character golden this fixture reproduces keeps its measured length.
        lines += [
            "",
            "## Combat",
            "",
            "- round: 3",
            "- active actor: sera-vane",
            "- order: rill, sera-vane",
            "- combatant: rill side=pc range=nearby actions=1/2",
            "- combatant: sera-vane side=npc range=close actions=2/2",
        ]
    lines += [
        "",
        "## Open hooks",
        "",
        "- The salt magistrate wants the writ back before dusk.",
        "- The ferryman expects payment for the last crossing.",
    ]
    lines += ["", "## Clocks", "", "- brackwater levy (3/6) hulls counted at the mole"]
    lines += [
        "",
        "## Unratified outcomes",
        "",
        "- [7] attribute_test success: Rill reads the tide line",
    ]
    lines += [
        "",
        "## Game-master-only facts",
        "",
        "<!-- Never quote this section to players. -->",
        "",
        "- The keeper already sold the writ's twin.",
        "- The crypt bell answers only to salt water.",
    ]
    lines += [
        f"- Secret {index:02d}: a hidden truth the players have not uncovered."
        for index in range(1, hidden_count + 1)
    ]
    lines += ["", "### Unratified GM-only outcomes", "", "- None recorded."]
    lines += [""]
    return "\n".join(line.rstrip() for line in lines)


def _canon_root(tmp_path, fact_count: int, hidden_count: int = 0, combat: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        _scene_render(fact_count, hidden_count, combat), encoding="utf-8"
    )
    (tmp_path / "world" / "locations").mkdir(parents=True)
    (tmp_path / "world" / "locations" / "the-eel-market.md").write_text(
        "## Public description\n\nStalls on the mud.\n\n## Hidden truths\n\nThe keeper lies.\n",
        encoding="utf-8",
    )
    (tmp_path / "world" / "npcs").mkdir()
    (tmp_path / "world" / "npcs" / "index.yaml").write_text(
        "npcs:\n  - id: sera-vane\n    name: Sera Vane\n    motive: recover the ledger\n",
        encoding="utf-8",
    )
    return tmp_path


def _states(digest) -> dict:
    return {section["name"]: section["state"] for section in digest.stats["scene_sections"]}
        # test_head_retention_alone_would_take_the_clocks_section asserts the
        # geometry both sums encode, at 19 facts and at 60, so the fallback stays
        # pinned without re-encoding the sums here.


# -- request payload composition ---------------------------------------------


def _request(*texts: str, system: str = "SYSTEM", tools: list | None = None) -> dict:
    """One request in the shape ``OpenAIModel.format_request`` assembles.

    The user messages carry typed text blocks and the system prompt carries a plain
    string, which is the exact asymmetry ``narrator.payload`` has to read. The narrator
    environment pins this shape against the real formatter in
    ``tests_narrator/test_model_payload.py``; this file pins the measurement.
    """
    messages = [{"role": "system", "content": system}]
    messages += [
        {"role": "user", "content": [{"type": "text", "text": text}]} for text in texts
    ]
    return {"messages": messages, "model": "test", "stream": True, "tools": tools or []}


def _digest_turn(marker: str, channel: str = "Rill: hello") -> tuple[str, str]:
    """One turn message carrying a digest, and the digest text it carries."""
    from narrator.canon import DIGEST_HEADING

    digest = f"{DIGEST_HEADING}\n\n{marker}"
    return prompt.turn_prompt(channel, canon=digest), digest


# -- superseded canon removal ------------------------------------------------


def _history(*texts: str) -> list[dict]:
    """Message history in the shape Strands keeps it, before OpenAI formatting."""
    return [{"role": "user", "content": [{"text": text}]} for text in texts]


def _seed_ruling(campaign_root: Path, ruling: dict) -> None:
    campaign = campaign_root / "campaign"
    campaign.mkdir(exist_ok=True)
    (campaign / "state.json").write_text(
        json.dumps({"fiction_debt": [], "pending_rulings": [ruling]}), encoding="utf-8"
    )


# -- planner public-scope digest (disclosure boundary) -----------------------


def _disclosure_root(tmp_path):
    """A campaign whose location and NPC index carry secrets the party has not uncovered."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(_scene_render(4, hidden_count=2), encoding="utf-8")
    (tmp_path / "world" / "locations").mkdir(parents=True)
    (tmp_path / "world" / "locations" / "the-eel-market.md").write_text(
        "---\n"
        "id: the-eel-market\n"
        "hidden_entities:\n  - the-listener\n"
        "---\n\n"
        "## Public description\n\nStalls on the mud at low water.\n\n"
        "## Immediate danger\n\nThe planks are slick.\n\n"
        "## Useful details\n\nRope and salt for sale.\n\n"
        "## Hidden truths\n\nThe keeper sold the writ's twin.\n\n"
        "## NPC motives\n\nSera Vane wants the ledger back.\n\n"
        "## Discoverable clues\n\nA bent brass clapper lies at the shrine.\n\n"
        "## Consequences\n\nThe Choir Below drowns the market at the next high tide.\n",
        encoding="utf-8",
    )
    (tmp_path / "world" / "npcs").mkdir()
    (tmp_path / "world" / "npcs" / "index.yaml").write_text(
        "npcs:\n"
        "  - id: sera-vane\n"
        "    name: Sera Vane\n"
        "    role: bell-keeper of the road shrine\n"
        "    faction: the-choir-below\n"
        "    motive: close the negotiation honestly\n"
        "    actions:\n      - strike a tuning note\n",
        encoding="utf-8",
    )
    return tmp_path


# -- M10: a decision answer is accepted when its meaning is unambiguous -------


def _decision_options(custom: bool = True):
    """The option list a clarification view carries, matching ``public_options``."""
    from narrator.decisions import PublicOption

    listed = (
        PublicOption(id="scramble", label="Desperate Scramble"),
        PublicOption(id="cautious", label="Cautious Search"),
        PublicOption(id="flee", label="Ignore and Flee"),
    )
    if not custom:
        return listed
    return listed + (
        PublicOption(id="own_approach", label="Describe your own approach", custom=True),
    )


# -- M11: printed mechanics match computed mechanics -------------------------


def _campaign_attributes() -> dict:
    """Minimal synthetic sheets for the malformed-announcement regression.

    These fixed targets exercise an invented target and an inverted verdict.
    Tests must not depend on an operator's mutable campaign files.
    """
    return {
        "rill": {"attributes": {"STR": 12, "DEX": 12, "CON": 11, "INT": 10, "WIS": 10, "CHA": 13}},
        "ossa": {"attributes": {"STR": 11, "DEX": 11, "CON": 9, "INT": 12, "WIS": 9, "CHA": 12}},
    }


class _CorrectionAgent:
    """One channel agent that answers the correction prompt with a fixed replacement."""

    def __init__(self, replacement: str) -> None:
        self._replacement = replacement
        self.prompts: list[str] = []

    async def invoke_async(self, prompt: str):
        self.prompts.append(prompt)
        return type(
            "Result", (), {"message": {"content": [{"text": self._replacement}]}}
        )()


def _engine_holding(facts: tuple) -> engine.NarratorEngine:
    """A bare engine carrying one turn's roll facts and nothing else.

    ``_correct_verdicts`` reads exactly one instance attribute, so this stays a
    standard-library test in a file that cannot install Strands (see this module's
    own docstring) while still driving the real method rather than a copy of it.
    """
    instance = engine.NarratorEngine.__new__(engine.NarratorEngine)
    instance._roll_facts_this_turn = list(facts)
    return instance


# -- first-mention items are established, not verified -----------------------


_FIRST_MENTION_EXAMPLE_PATTERN = re.compile(
    r"A first-mention item or container.*?```json\n(.*?)\n```", re.DOTALL
)


# -- a rest declaration is resolved through the rest tool, not refused -------


_REST_EXAMPLE_PATTERN = re.compile(
    r"A short rest declaration:.*?```json\n(.*?)\n```", re.DOTALL
)


_MENTION_NARRATION = "An iron pulley is bolted to the eastern wall."


def _roster(**overrides) -> dict:
    from narrator.sweep import empty_roster

    roster = empty_roster()
    roster["npcs"] = {"rade": "dead", "orso-pell": "alive"}
    roster["characters"] = {"ossa": "Ossa", "rill": "Rill"}
    roster["objects"] = {"tower-door": "barred"}
    roster["exits"] = ["the-road-shrine"]
    roster.update(overrides)
    return roster


_RESURRECTIONS = {
    "en": "Rade is alive beside a red cart.",
    "fr": "Rade est vivant près d’une charrette rouge.",
    "de": "Rade ist am Leben neben einem roten Wagen.",
    "ja": "ラデは生きていて、赤い荷車のそばにいる。",
    "ru": "Раде жив рядом с красной телегой.",
}


def _entity_campaign(root: Path) -> None:
    """A campaign whose roster carries one dead NPC, one object, two characters."""
    campaign = root / "campaign"
    (campaign / "characters").mkdir(parents=True, exist_ok=True)
    (campaign / "state.json").write_text(
        json.dumps(
            {
                "fiction_debt": [],
                "npcs": {
                    "rade": {"id": "rade", "name": "Rade", "level": 1, "hp": 0, "hp_max": 5,
                             "damage": 4, "status": "dead"},
                },
                "scene": {
                    "summary": "The party stands in the eel market at low water.",
                    "objects": {"tower-door": {"id": "tower-door", "state": "barred"}},
                    "exits": ["the-road-shrine"],
                    "persons": {},
                },
            }
        ),
        encoding="utf-8",
    )
    for character_id, name in (("ossa", "Ossa"), ("rill", "Rill")):
        (campaign / "characters" / f"{character_id}.json").write_text(
            json.dumps({"id": character_id, "name": name, "coins": 5, "hp": 9, "equipment": []}),
            encoding="utf-8",
        )


def _load_catalog(language: str):
    from narrator.locale import load

    return load(language, REPO_ROOT / "locale")


__all__ = [
    'json',
    're',
    'Path',
    'pytest',
    'REPO_ROOT',
    'delivery',
    'engine',
    'ledger',
    'locale',
    'policy',
    'prompt',
    'ChannelMessage',
    'InboundTurn',
    'NarratorConfig',
    'resolved_server_command',
    'server_env',
    '_ROMANCE_NOTICE',
    '_RecordingAdapter',
    '_StubResult',
    '_StubAgent',
    '_engine_with_stub',
    '_seed_ledger',
    '_full_debt',
    '_character_entry',
    '_npc_entry',
    '_write_character',
    '_scene_render',
    '_canon_root',
    '_states',
    '_request',
    '_digest_turn',
    '_history',
    '_seed_ruling',
    '_disclosure_root',
    '_decision_options',
    '_campaign_attributes',
    '_CorrectionAgent',
    '_engine_holding',
    '_FIRST_MENTION_EXAMPLE_PATTERN',
    '_REST_EXAMPLE_PATTERN',
    '_MENTION_NARRATION',
    '_roster',
    '_RESURRECTIONS',
    '_entity_campaign',
    '_load_catalog',
]
