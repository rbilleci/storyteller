"""Pin the engine-performed combat recovery behind the ResolutionGuard withhold.

These tests drive the recovery against a REAL ``bsh_mcp.GameService`` behind a fake
client, so every claim below is about actual audited state changes, not stubs: the
attack event lands in the audit log, the enemy's hit points move, the guard clears.
The ``ScriptedRoller`` is a local copy of ``tests/conftest.py``'s, kept here rather
than imported because the two test trees import nothing from each other anywhere in
this repository.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from engine_fixtures import _seed_ledger

from bsh_mcp.dice import Roller
from bsh_mcp.service import GameService
from narrator.config import NarratorConfig
from narrator.decisions import (
    DecisionSubmission,
    resolution_from_submission,
    risk_confirmation,
)
from narrator.engine import NarratorEngine
from narrator.resolution_guard import ResolutionGuard

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ScriptedRoller(Roller):
    script: list[int] = field(default_factory=list)

    def die(self, sides: int) -> int:
        if self.script:
            value = self.script.pop(0)
            if not 1 <= value <= sides:
                raise AssertionError(f"scripted value {value} cannot appear on a d{sides}")
            return value
        return super().die(sides)

    def queue(self, *values: int) -> ScriptedRoller:
        self.script.extend(values)
        return self


class _ServiceClient:
    """Forward engine tool calls to a real GameService, shaped like an MCP result."""

    def __init__(self, service: GameService) -> None:
        self._service = service
        self.calls: list[str] = []

    def call_tool_sync(self, tool_use_id: str, name: str, arguments: dict) -> dict:
        self.calls.append(name)
        result = getattr(self._service, name)(**arguments)
        return {
            "status": "success" if result.get("ok") else "error",
            "content": [{"text": json.dumps(result)}],
        }


def _service(root: Path, roller: ScriptedRoller) -> GameService:
    shutil.copytree(REPO_ROOT / "rules", root / "rules")
    shutil.copytree(REPO_ROOT / "world", root / "world")
    service = GameService(root, roller=roller)
    service.store.initialize(title="Recovery test")
    for total in (12, 12, 12, 12, 12, 12):
        roller.queue(6, 6)
    created = service.character_create(
        discord_user_id="test-player",
        name="Rill",
        origin="barbarian",
        # No scout: its initiative Advantage rolls a second d20 the script
        # does not control. These three keep every roll single-die.
        backgrounds=["hunter", "survivor", "raider"],
        weapons=["long knife"],
    )
    assert created["ok"], created
    return service


def _open_fight(
    service: GameService, roller: ScriptedRoller, *, initiative: int, band: str
) -> None:
    npc = service.npc_create(name="Rade", level=1, motive="survive")
    assert npc["ok"], npc
    roller.queue(initiative)
    started = service.combat_start(
        pc_ids=["rill"], npc_ids=["rade"], initial_ranges={"rade": band}, reason="test"
    )
    assert started["ok"], started


def _engine_with(service: GameService, root: Path) -> NarratorEngine:
    engine = NarratorEngine(NarratorConfig(campaign_root=root))
    engine._client = _ServiceClient(service)  # noqa: SLF001 - test seam
    return engine


def _confirmed_attack_guard() -> ResolutionGuard:
    resolution = resolution_from_submission(
        risk_confirmation("violence"),
        "rill",
        DecisionSubmission(presentation_token="x" * 16, selection_id="confirm"),
    )
    return ResolutionGuard((resolution,))


def _bound_defence_guard(method: str = "dodge") -> ResolutionGuard:
    from narrator.decisions import CombatDefendDirective, DecisionResolution

    resolution = DecisionResolution(
        character_id="rill",
        kind="approach",
        answer_kind="option",
        selection_id="defend",
        directive=CombatDefendDirective(
            kind="combat_defend", character_id="rill", method=method
        ),
    )
    return ResolutionGuard((resolution,))


def test_a_forgotten_confirmed_attack_is_rolled_by_the_engine(tmp_path: Path):
    """Test a forgotten confirmed attack is rolled by the engine.
    """
    roller = ScriptedRoller()
    service = _service(tmp_path, roller)
    _open_fight(service, roller, initiative=5, band="close")  # rill acts first

    engine = _engine_with(service, tmp_path)
    guard = _confirmed_attack_guard()
    assert guard.unused_error() == "the confirmed action rolled no resolving mechanic"

    roller.queue(3, 4)  # the engine's attack: a hit, then its damage die
    lines = engine._recover_unrolled_combat(guard)  # noqa: SLF001

    assert guard.unused_error() == ""
    assert any("hits Rade" in line for line in lines), lines
    events = [e for e in service.store.read_events(limit=10) if e["tool"] == "combat_attack"]
    assert len(events) == 1
    assert events[0]["outcome"] == "success"
    assert service.store.read_state().npcs["rade"].hp == 1  # 5 - 4 rolled damage
    # The roll fact is recorded exactly as a model-issued call's would be, so the
    # downstream announcement injection states the engine's roll.
    facts = engine._roll_facts_this_turn  # noqa: SLF001
    assert any(f["tool"] == "combat_attack" and f["character"] == "rill" for f in facts)


def test_the_engine_closes_range_before_its_attack(tmp_path: Path):
    roller = ScriptedRoller()
    service = _service(tmp_path, roller)
    _open_fight(service, roller, initiative=5, band="nearby")

    engine = _engine_with(service, tmp_path)
    guard = _confirmed_attack_guard()
    roller.queue(3, 4)  # attack roll and damage, after the (roll-less) move
    lines = engine._recover_unrolled_combat(guard)  # noqa: SLF001

    assert guard.unused_error() == ""
    assert any("closes to close range" in line for line in lines), lines
    tools = [e["tool"] for e in service.store.read_events(limit=10)]
    assert "combat_move" in tools and "combat_attack" in tools


def test_spending_every_action_closing_range_defers_the_attack(tmp_path: Path):
    """Distant with two actions: two real moves, no attack possible this turn.

    The hazard is satisfied by the engaged deferral -- the same rules-made-it-
    impossible reasoning the guard grants a combat_start whose initiative gave the
    opposition the first move -- and the turn has advanced to the enemy.
    """
    roller = ScriptedRoller()
    service = _service(tmp_path, roller)
    _open_fight(service, roller, initiative=5, band="distant")

    engine = _engine_with(service, tmp_path)
    guard = _confirmed_attack_guard()
    lines = engine._recover_unrolled_combat(guard)  # noqa: SLF001

    assert guard.unused_error() == ""
    assert len([line for line in lines if "closes to" in line]) == 2
    tools = [e["tool"] for e in service.store.read_events(limit=10)]
    assert "combat_attack" not in tools
    state = service.store.read_state()
    assert state.combat.active_actor == "rade"  # both actions spent; turn advanced


def test_several_living_enemies_leave_the_withhold_standing(tmp_path: Path):
    """An ambiguous target is the model's to choose, never the engine's."""
    roller = ScriptedRoller()
    service = _service(tmp_path, roller)
    first = service.npc_create(name="Rade", level=1, motive="survive")
    second = service.npc_create(name="Reed Thug", level=1, motive="rob")
    roller.queue(5)
    started = service.combat_start(
        pc_ids=["rill"],
        npc_ids=[first["npc_id"], second["npc_id"]],
        initial_ranges={first["npc_id"]: "close", second["npc_id"]: "close"},
    )
    assert started["ok"], started

    engine = _engine_with(service, tmp_path)
    guard = _confirmed_attack_guard()
    lines = engine._recover_unrolled_combat(guard)  # noqa: SLF001

    assert lines == []
    assert guard.unused_error() == "the confirmed action rolled no resolving mechanic"
    tools = [e["tool"] for e in service.store.read_events(limit=10)]
    assert "combat_attack" not in tools


def test_an_enemy_first_fight_leaves_the_hazard_to_its_own_turn(tmp_path: Path):
    """When the opposition holds the open turn, the attack recovery does nothing.

    (In production this hazard is already satisfied by combat_start's deferral;
    this pins that the recovery never rolls a player attack out of turn even when
    a guard somehow still holds the hazard open.)
    """
    roller = ScriptedRoller()
    service = _service(tmp_path, roller)
    _open_fight(service, roller, initiative=18, band="close")  # rade acts first

    engine = _engine_with(service, tmp_path)
    guard = _confirmed_attack_guard()
    lines = engine._recover_unrolled_combat(guard)  # noqa: SLF001

    assert lines == []
    assert guard.unused_error() == "the confirmed action rolled no resolving mechanic"


def test_a_forgotten_bound_defence_is_rolled_by_the_engine(tmp_path: Path):
    """Test a forgotten bound defence is rolled by the engine.
    """
    roller = ScriptedRoller()
    service = _service(tmp_path, roller)
    _open_fight(service, roller, initiative=18, band="close")  # rade acts first

    engine = _engine_with(service, tmp_path)
    guard = _bound_defence_guard("dodge")
    assert guard.unused_error() == "the selected decision's required mechanic was not used"

    roller.queue(18)  # a failed dodge: damage lands, which proves the real roll ran
    lines = engine._recover_unrolled_combat(guard)  # noqa: SLF001

    assert guard.unused_error() == ""
    assert any("fails to dodge" in line for line in lines), lines
    state = service.store.read_state()
    assert state.combat.actors["rade"].actions_used == 1  # the enemy's action is spent
    assert state.combat.active_actor == "rill"  # and its turn advanced itself
    facts = engine._roll_facts_this_turn  # noqa: SLF001
    assert any(f["tool"] == "combat_defend" and f["character"] == "rill" for f in facts)


class _StubResult:
    def __init__(self, text: str) -> None:
        self.message = {"content": [{"text": text}]}


class _RenarratingAgent:
    """First reply: the stale mid-turn draft; second reply: the re-narration.
    """

    def __init__(self, first: str, second: str, *, second_raises: bool = False) -> None:
        self.replies = [first, second]
        self.calls: list[str] = []
        self.messages: list[dict] = []
        self.second_raises = second_raises
        self.freeze_at_invoke: list[str] = []
        self.engine = None

    async def invoke_async(self, text: str):
        self.calls.append(text)
        if self.engine is not None:
            self.freeze_at_invoke.append(self.engine._corrective_tool_freeze)  # noqa: SLF001
        if len(self.calls) > 1 and self.second_raises:
            raise RuntimeError("model endpoint fault")
        reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
        self.messages.append({"role": "user", "content": [{"text": text}]})
        self.messages.append({"role": "assistant", "content": [{"text": reply}]})
        return _StubResult(reply)


def _turn() -> InboundTurn:  # noqa: F821 -- import is local, kept out of module scope
    from narrator.channels.base import ChannelMessage, InboundTurn

    return InboundTurn(
        channel_id="c1",
        mention=ChannelMessage("Rill", "I attack the fishmonger with my knife"),
    )


def _hazard_resolution(character_id: str = "rill"):
    from narrator.decisions import DecisionResolution, HazardResolutionDirective

    return DecisionResolution(
        character_id=character_id,
        kind="confirmation",
        answer_kind="confirm",
        selection_id="confirm",
        directive=HazardResolutionDirective(
            kind="hazard_resolution", character_id=character_id, category="violence"
        ),
    )


_STALE_DRAFT = "Rade is out of reach. Do you spend an action to close the distance?"
_RECOVERY_LINES = [
    "rill closes to close range with Rade.",
    "Rill hits Rade for 6 damage. Rade is dead.",
]


def _engine_with_scripted_recovery(tmp_path: Path, agent) -> NarratorEngine:
    _seed_ledger(tmp_path, [])
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    agent.engine = engine
    engine._agent_for = lambda channel_id: agent  # noqa: SLF001 - test seam

    def scripted_recovery(guard):
        guard.satisfy_hazard("rill")
        return list(_RECOVERY_LINES)

    engine._recover_unrolled_combat = scripted_recovery  # noqa: SLF001 - test seam
    return engine


async def test_a_recovered_turn_renarrates_and_supersedes_the_stale_draft(tmp_path: Path):
    """Test a recovered turn renarrates and supersedes the stale draft.
    """
    fiction = "You surge forward; the long knife takes Rade under the ribs. He folds, dead."
    agent = _RenarratingAgent(_STALE_DRAFT, fiction)
    engine = _engine_with_scripted_recovery(tmp_path, agent)

    outcome = await engine.run_turn(
        _turn(), decision_resolutions=(_hazard_resolution(),), decision_action_fingerprint=""
    )

    assert outcome.withheld is False
    assert outcome.narration == fiction
    assert "Do you spend an action" not in outcome.narration
    # The re-invoke both produces the fiction and writes the recovered facts into
    # the agent's own history -- the world-model fork D3 traced.
    assert len(agent.calls) == 2
    assert all(line in agent.calls[1] for line in _RECOVERY_LINES)
    assert agent.freeze_at_invoke[1]  # combat tools frozen while the re-invoke ran
    assert engine._corrective_tool_freeze == ""  # noqa: SLF001 - and thawed after
    assert len(engine._recovery_log) == 1  # noqa: SLF001 - one entry per model turn
    record = engine._recovery_log[-1]  # noqa: SLF001
    assert record["triggered"] and record["retried"]
    assert record["resolved"] is True
    assert record["fell_back"] is False


async def test_a_faulted_renarration_falls_back_to_the_engine_lines(tmp_path: Path):
    """A re-invoke fault must never cost the table the mechanics: the pre-R1
    delivery shape (draft plus engine-authored lines) is the fallback."""
    agent = _RenarratingAgent(_STALE_DRAFT, "", second_raises=True)
    engine = _engine_with_scripted_recovery(tmp_path, agent)

    outcome = await engine.run_turn(
        _turn(), decision_resolutions=(_hazard_resolution(),), decision_action_fingerprint=""
    )

    assert outcome.withheld is False
    assert outcome.narration == _STALE_DRAFT + "\n\n" + "\n".join(_RECOVERY_LINES)
    assert engine._corrective_tool_freeze == ""  # noqa: SLF001 - thawed on the fault path too
    record = engine._recovery_log[-1]  # noqa: SLF001
    assert record["fell_back"] is True
    assert record["resolved"] is False


async def test_a_blank_renarration_falls_back_to_the_engine_lines(tmp_path: Path):
    agent = _RenarratingAgent(_STALE_DRAFT, "   ")
    engine = _engine_with_scripted_recovery(tmp_path, agent)

    outcome = await engine.run_turn(
        _turn(), decision_resolutions=(_hazard_resolution(),), decision_action_fingerprint=""
    )

    assert outcome.narration == _STALE_DRAFT + "\n\n" + "\n".join(_RECOVERY_LINES)
    record = engine._recovery_log[-1]  # noqa: SLF001
    assert record["retried"] is True
    assert record["fell_back"] is True


def test_the_corrective_freeze_rejects_only_combat_tools(tmp_path: Path):
    """The freeze is scoped: during a re-narration every ``combat_*`` call is
    refused (a re-invoked model must not roll a second attack), and everything
    else stays subject only to the ordinary ceiling and guard checks."""
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    assert engine._corrective_freeze_refusal("combat_attack") == ""  # noqa: SLF001 - freeze off
    engine._corrective_tool_freeze = "narrate from the stated results"  # noqa: SLF001
    assert engine._corrective_freeze_refusal("combat_attack") != ""  # noqa: SLF001
    assert engine._corrective_freeze_refusal("combat_move") != ""  # noqa: SLF001
    assert engine._corrective_freeze_refusal("scene_commit") == ""  # noqa: SLF001
    assert engine._corrective_freeze_refusal("campaign_status") == ""  # noqa: SLF001


def test_a_bound_defence_outside_the_enemys_turn_stays_withheld(tmp_path: Path):
    """No open enemy turn means no attack to defend against; the withhold stands."""
    roller = ScriptedRoller()
    service = _service(tmp_path, roller)
    _open_fight(service, roller, initiative=5, band="close")  # rill acts first

    engine = _engine_with(service, tmp_path)
    guard = _bound_defence_guard("dodge")
    lines = engine._recover_unrolled_combat(guard)  # noqa: SLF001

    assert lines == []
    assert (
        guard.unused_error() == "the selected decision's required mechanic was not used"
    )
    tools = [e["tool"] for e in service.store.read_events(limit=10)]
    assert "combat_defend" not in tools
