"""The player-facing tool surface, frozen and asserted before any channel starts.

Strands inverts the default. An ``Agent`` receives tools only from sources the caller
names, so nothing arrives unasked. This module makes that guarantee explicit rather than
incidental: it states the exact set the narrator may expose and refuses to start on any
difference, in either direction.

Set equality is the load-bearing property, not a subset check. A subset check would let
a future Strands release auto-register a tool and still pass, which is precisely the
silent-enable class the Hermes ledgers could not close.

This module imports nothing outside the standard library, so the main test environment
can pin it without installing Strands. ``tests_narrator/test_policy_live.py`` asserts the
same equality against a live Model Context Protocol (MCP) server and a real agent.
"""

from __future__ import annotations

#: The tools the model may call. Kept identical to the model-facing subset of
#: ``EXPECTED_TOOLS`` in ``tests/test_mcp_wiring.py``, which pins the server's registration.
MCP_TOOLS: frozenset[str] = frozenset(
    {
        "campaign_status",
        "character_options",
        "character_create",
        "character_advance",
        "character_sheet",
        "attribute_test",
        "group_test",
        "usage_roll",
        "doom_roll",
        "npc_create",
        "combat_start",
        "combat_begin_turn",
        "combat_move",
        "combat_attack",
        "combat_defend",
        "combat_end_turn",
        "combat_close",
        "rest",
        "helpless_roll",
        "grant_runic_weapon",
        "use_ability",
        "inventory_update",
        "scene_commit",
        "session_close",
    }
)

#: Tools the server publishes for the engine alone. The settle step calls
#: ``ledger_settle`` and the adjudicate step calls ``ability_apply_ruling`` through
#: the engine's own Model Context Protocol client. The model must never reach either:
#: a prompt-injected waive would erase pending outcomes, and a prompt-injected ruling
#: would resolve a demon's theft on an attacker's say-so. The fix for an unreliable
#: model must not hand that model a new capability.
ENGINE_ONLY_TOOLS: frozenset[str] = frozenset({"ledger_settle", "ability_apply_ruling"})

#: Everything the server must serve: the model-facing ``MCP_TOOLS`` plus ``ENGINE_ONLY_TOOLS``.
#: Equality against this set is asserted at startup, so a server that stops serving
#: an engine-only tool fails loudly instead of leaving the settle or adjudicate step broken.
MCP_SERVED_TOOLS: frozenset[str] = MCP_TOOLS | ENGINE_ONLY_TOOLS

#: The activation tool the ``AgentSkills`` plugin contributes. It loads a skill body on
#: demand and reads nothing outside the skill directories the engine names.
SKILLS_TOOL: frozenset[str] = frozenset({"skills"})

#: Everything the narrator may expose to players. Nothing else, ever. ``ledger_settle``
#: is deliberately absent: served and model-facing are two different surfaces, and
#: ``assert_tool_surface`` holds the model-facing one to equality.
PLAYER_FACING_TOOLS: frozenset[str] = MCP_TOOLS | SKILLS_TOOL

#: The tools a game-master-discussion turn may call: pure reads, plus the skills
#: loader. A turn the player addressed with the configured game-master designator
#: (``NarratorConfig.gm_address``) is out-of-fiction conversation about the game, and
#: its guarantee is that nothing mechanical can happen: no die can roll, no state can
#: change, no scene entry can be written. ``NarratorEngine`` enforces this with the
#: same ``BeforeToolCallEvent`` cancellation the per-turn ceiling uses, allowing only
#: this set; the guarantee is structural, not a prompt instruction, for the reason
#: ``ENGINE_ONLY_TOOLS`` records. Everything here answers from the record --
#: ``attribute_test`` and the other roll tools are deliberately absent because a roll
#: is a mechanic even when no state field moves, and ``scene_commit`` is absent
#: because out-of-fiction talk must never become canon.
GM_DISCUSSION_TOOLS: frozenset[str] = (
    frozenset({"campaign_status", "character_options", "character_sheet"}) | SKILLS_TOOL
)

#: Tools Strands ships in ``strands/vended_tools/`` that must never reach a player.
#: Listing them turns a silent regression into a named one: the equality check already
#: rejects any of these, and this canary reports which one arrived.
FORBIDDEN_TOOLS: frozenset[str] = frozenset(
    {"bash", "file_editor", "http_request", "shell", "python_repl", "editor"}
)


class ToolSurfaceError(RuntimeError):
    """Raised when the agent's tool surface differs from the frozen manifest."""


def assert_tool_surface(actual: object) -> None:
    """Refuse to start unless the agent exposes exactly ``PLAYER_FACING_TOOLS``.

    ``actual`` accepts any iterable of tool names, because Strands reports them as a
    list and tests supply sets.

    ``NarratorEngine.start`` calls this against a discarded probe agent before the
    service iterates any adapter, and ``_build_agent`` calls it again for every agent a
    channel receives. A surface violation therefore stops the process rather than
    reaching a player, whether or not a channel ever yields a turn.
    """
    names = frozenset(actual)
    if names == PLAYER_FACING_TOOLS:
        return

    unexpected = names - PLAYER_FACING_TOOLS
    missing = PLAYER_FACING_TOOLS - names
    forbidden = unexpected & FORBIDDEN_TOOLS

    detail = []
    if forbidden:
        detail.append(f"forbidden tools present: {sorted(forbidden)}")
    if unexpected - forbidden:
        detail.append(f"unexpected tools present: {sorted(unexpected - forbidden)}")
    if missing:
        detail.append(f"expected tools missing: {sorted(missing)}")
    raise ToolSurfaceError("; ".join(detail))
