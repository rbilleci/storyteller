"""Model Context Protocol wiring tests: served tool surface, protocol envelope,
resource reads, and argument repair."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import pytest
from conftest import REPO_ROOT

from bsh_mcp.server import ARGUMENT_SHAPES, create_server, repair_arguments
from bsh_mcp.service import GameService
from bsh_mcp.store import CampaignStore

#: The tools the Model Context Protocol server publishes. `src/narrator/policy.py`
#: splits them into two surfaces: the model-facing names frozen as `MCP_TOOLS`, plus
#: `ledger_settle` and `ability_apply_ruling` in `ENGINE_ONLY_TOOLS`, which only the
#: narrator engine may call. Keep this list identical to `MCP_SERVED_TOOLS`: this one
#: proves the server serves them, and the policy proves nothing beyond the model-facing
#: `MCP_TOOLS` reaches a player.
EXPECTED_TOOLS = {
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
    "ledger_settle",
    "ability_apply_ruling",
}


# -- Model Context Protocol wiring -------------------------------------------


async def test_the_server_exposes_exactly_the_whitelisted_tools(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == EXPECTED_TOOLS
    runic = next(tool for tool in tools if tool.name == "grant_runic_weapon")
    assert "weapon_int" not in (runic.input_schema or {}).get("properties", {})


async def test_every_tool_carries_an_operational_description(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    for tool in await server.list_tools():
        assert tool.description and len(tool.description) > 40, tool.name


async def test_the_server_exposes_the_campaign_resources(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    resources = {str(resource.uri) for resource in await server.list_resources()}
    templates = {
        template.uri_template for template in await server.list_resource_templates()
    }
    assert "bsh://rules/quick-reference" in resources
    assert "bsh://rules/subsystems" in resources
    assert "bsh://rules/effects" in resources
    assert "bsh://campaign/status" in resources
    assert "bsh://campaign/current-scene" in resources
    assert "bsh://campaign/characters" in resources
    assert "bsh://campaign/character/{character_id}" in templates
    assert "bsh://world/location/{location_id}" in templates


def tool_payload(result) -> dict:
    """Decode the JavaScript Object Notation (JSON) body of a tool result."""
    return json.loads(result.content[0].text)


async def test_calling_a_tool_over_the_protocol_returns_the_envelope(campaign_root: Path):
    CampaignStore(campaign_root).initialize(title="Protocol Test")
    server = create_server(campaign_root, seed=1)

    payload = tool_payload(await server.call_tool("campaign_status", {}))
    assert payload["ok"] is True
    assert payload["campaign"]["title"] == "Protocol Test"


async def test_a_protocol_error_returns_the_structured_envelope(campaign_root: Path):
    CampaignStore(campaign_root).initialize(title="Protocol Test")
    server = create_server(campaign_root, seed=1)

    payload = tool_payload(await server.call_tool("character_sheet", {"character_id": "nobody"}))
    assert payload["ok"] is False
    assert payload["error"] == "character_not_found"
    assert payload["allowed_next_steps"]


async def test_a_full_turn_runs_over_the_protocol(campaign_root: Path):
    CampaignStore(campaign_root).initialize(title="Protocol Test")
    server = create_server(campaign_root, seed=99)

    created = tool_payload(
        await server.call_tool(
            "character_create",
            {
                "discord_user_id": "1001",
                "name": "Mara",
                "origin": "barbarian",
                "backgrounds": ["hunter", "survivor", "raider"],
                "weapons": ["long knife"],
            },
        )
    )
    assert created["ok"] is True

    tested = tool_payload(
        await server.call_tool(
            "attribute_test",
            {
                "character_id": "mara",
                "attribute": "DEX",
                "reason": "cross the open mud unseen",
                "stakes_success": "Mara reaches the shrine wall unseen.",
                "stakes_failure": "The patrol marks movement on the mud.",
                "advantage": True,
                "opponent_level": 2,
            },
        )
    )
    assert tested["ok"] is True
    assert tested["roll"]["dice"]
    assert tested["threat_modifier"] == 1

    committed = tool_payload(
        await server.call_tool(
            "scene_commit",
            {
                "public_summary": "Mara reaches the shrine wall.",
                "location_id": "the-road-shrine",
                "in_game_time_delta_minutes": 5,
            },
        )
    )
    assert committed["ok"] is True
    assert CampaignStore(campaign_root).read_state().scene.location_id == "the-road-shrine"


def schema_shape(schema: dict) -> str | None:
    """Return array, object, or None for one property schema, unwrapping null unions."""
    candidates = [schema] + [
        variant for variant in schema.get("anyOf", []) if isinstance(variant, dict)
    ]
    for candidate in candidates:
        if candidate.get("type") in ("array", "object"):
            return candidate["type"]
    return None


async def test_the_argument_repair_map_matches_the_generated_schemas(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    for tool in await server.list_tools():
        properties = (tool.input_schema or {}).get("properties", {})
        expected = {
            name: shape
            for name, schema in properties.items()
            if (shape := schema_shape(schema)) is not None
        }
        assert ARGUMENT_SHAPES.get(tool.name, {}) == expected, tool.name


_KNOWN_UNREACHABLE_SERVICE_PARAMETERS: dict[str, set[str]] = {
    "scene_commit": {"object_updates"},
}


async def test_every_service_parameter_is_reachable_through_its_served_tool(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    for tool in await server.list_tools():
        method = getattr(GameService, tool.name, None)
        if method is None:
            continue  # a resource-backed name with no matching GameService method
        service_params = set(inspect.signature(method).parameters) - {"self"}
        exposed_params = set((tool.input_schema or {}).get("properties", {}))
        missing = service_params - exposed_params - _KNOWN_UNREACHABLE_SERVICE_PARAMETERS.get(
            tool.name, set()
        )
        assert not missing, (
            f"{tool.name}: GameService accepts {sorted(missing)}, "
            "but the served tool's schema does not expose it"
        )


@pytest.mark.parametrize(
    "tool_name,arguments,repaired",
    [
        (
            "scene_commit",
            {"visible_changes": "one fact", "clock_updates": []},
            {"visible_changes": ["one fact"], "clock_updates": {}},
        ),
        (
            "group_test",
            {"character_ids": "mara", "attribute": "DEX"},
            {"character_ids": ["mara"], "attribute": "DEX"},
        ),
        ("campaign_status", {"unmapped": "value"}, {"unmapped": "value"}),
        (
            "rest",
            {"character_ids": ["mara"], "rest_type": "short"},
            {"character_ids": ["mara"], "rest_type": "short"},
        ),
    ],
)
def test_argument_repair_fixes_only_the_two_known_shapes(
    tool_name: str, arguments: dict, repaired: dict
):
    assert repair_arguments(tool_name, arguments)[0] == repaired


async def test_a_string_argument_is_repaired_over_a_live_stdio_session(campaign_root: Path):
    """Exercise the middleware through a real protocol connection.

    ``MCPServer.call_tool`` bypasses the dispatcher, so only a genuine session
    proves the repair reaches production traffic.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    CampaignStore(campaign_root).initialize(title="Repair Test")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "src" / "bsh_mcp" / "server.py")],
        env={**os.environ, "BSH_CAMPAIGN_ROOT": str(campaign_root)},
        cwd=str(REPO_ROOT),
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "scene_commit",
                {
                    "public_summary": "The tide turns.",
                    "visible_changes": "The plank walk floods to the knee.",
                    "clock_updates": [],
                },
            )
            payload = json.loads(
                "".join(item.text for item in result.content if hasattr(item, "text"))
            )

    assert payload["ok"] is True
    scene = CampaignStore(campaign_root).read_state().scene
    assert scene.visible_facts == ["The plank walk floods to the knee."]


async def test_reading_a_resource_over_the_protocol(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    contents = await server.read_resource("bsh://rules/quick-reference")
    body = "".join(item.content for item in contents)
    assert "Attribute Tests" in body


async def test_reading_the_subsystems_resource_over_the_protocol(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    contents = await server.read_resource("bsh://rules/subsystems")
    payload = json.loads("".join(item.content for item in contents))
    assert len(payload["demonic_pacts"]["demons"]) == 13
    assert payload["sorcery"]["spell_table"]["die"] == "d100"


async def test_reading_the_effects_resource_over_the_protocol(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    contents = await server.read_resource("bsh://rules/effects")
    payload = json.loads("".join(item.content for item in contents))
    assert payload["effects"]["berserker_rage"]["activation"] == "toggle"


async def test_reading_a_world_location_over_the_protocol(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    contents = await server.read_resource("bsh://world/location/the-river-cave")
    payload = json.loads("".join(item.content for item in contents))
    assert payload["meta"]["id"] == "the-river-cave"
    assert "Hidden truths" in payload["body"]
