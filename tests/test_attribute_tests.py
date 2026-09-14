"""Attribute test and usage die tool tests."""

from __future__ import annotations

from conftest import make_character
from tools_shared import make_fighter

from bsh_mcp.dice import DEPLETED
from bsh_mcp.service import GameService

# -- attribute tests ---------------------------------------------------------


def test_attribute_test_applies_threat_level(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(12)
    result = service.attribute_test("mara", "STR", "wrestle the pilot", "the way opens", "the cost lands", opponent_level=3)
    assert result["threat_modifier"] == 2
    assert result["roll"]["total"] == 14
    assert result["outcome"] == "failure"


def test_critical_failure_rolls_doom_automatically(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(20, 1)
    result = service.attribute_test("mara", "WIS", "listen at the crypt door", "the way opens", "the cost lands")
    assert result["outcome"] == "critical_failure"
    assert result["doom"]["current_die"] == "d4"
    assert service.store.read_character("mara").doom_die == "d4"


def test_calling_on_doom_subtracts_the_result_and_steps_the_die_down(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5, 15)
    result = service.attribute_test(
        "mara", "STR", "hold the gate shut", "the way opens", "the cost lands", call_on_doom=True
    )
    assert result["called_on_doom"]["current_die"] == "d4"
    assert result["roll"]["total"] == 10
    assert result["outcome"] == "success"


def test_a_spent_doom_die_cannot_be_called_upon(service: GameService, roller):
    make_fighter(service, roller)
    service.doom_roll("mara", "restore")
    for _ in range(2):
        roller.queue(1)
        service.doom_roll("mara", "burn it down", mode="call_on_doom")
    result = service.attribute_test("mara", "STR", "one last heave", "the way opens", "the cost lands", call_on_doom=True)
    assert result["ok"] is False
    assert result["error"] == "doom_depleted"


def test_a_depleted_doom_die_makes_every_test_disadvantaged(service: GameService, roller):
    make_fighter(service, roller)
    for _ in range(2):
        roller.queue(1)
        service.doom_roll("mara", "spend it", mode="call_on_doom")
    assert service.store.read_character("mara").doom_die == DEPLETED

    roller.queue(3, 17)
    result = service.attribute_test("mara", "STR", "shove the door", "the way opens", "the cost lands")
    assert result["roll"]["dice"] == [3, 17]
    assert result["roll"]["selected"] == 17
    assert any("Doomed" in warning for warning in result["warnings"])


def test_group_test_succeeds_when_half_the_party_succeeds(service: GameService, roller):
    make_fighter(service, roller)
    make_character(
        service,
        roller,
        name="Ulf",
        origin="civilised",
        backgrounds=("bodyguard", "legionnaire", "sword-master"),
        weapons=("arming sword",),
    )
    roller.queue(4, 18)
    result = service.group_test(["mara", "ulf"], "DEX", "slip past the customs guard", "the group slips through", "the group is marked")
    assert result["successes"] == 1
    assert result["outcome"] == "success"
    assert len(result["individual"]) == 2


def test_street_urchin_stealth_test_rolls_with_advantage(service: GameService, roller):
    """Test street urchin stealth test rolls with advantage.
    """
    make_character(
        service, roller, name="Mara", origin="civilised",
        backgrounds=("street-urchin", "bookworm", "diplomat"),
    )
    roller.queue(15, 3)
    result = service.attribute_test(
        "mara", "DEX", "slip past the watch", "the way opens", "the cost lands",
        category="stealth",
    )
    assert result["ok"], result
    assert result["roll"]["edge"] == "advantage"
    assert result["category"] == "stealth"


def test_category_without_the_background_stays_single(service: GameService, roller):
    make_fighter(service, roller)  # no street-urchin
    roller.queue(10)
    result = service.attribute_test(
        "mara", "DEX", "slip past the watch", "the way opens", "the cost lands",
        category="stealth",
    )
    assert result["ok"], result
    assert result["roll"]["edge"] == "single"


def test_unknown_test_category_is_refused_before_rolling(service: GameService, roller):
    make_fighter(service, roller)
    before = service.store.read_state().event_seq
    result = service.attribute_test(
        "mara", "DEX", "slip past the watch", "the way opens", "the cost lands",
        category="sneaking",
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_test_category"
    assert service.store.read_state().event_seq == before


def test_group_test_category_edges_only_the_urchin(service: GameService, roller):
    make_character(
        service, roller, name="Mara", origin="civilised",
        backgrounds=("street-urchin", "bookworm", "diplomat"),
    )
    make_character(
        service, roller, name="Ulf", origin="barbarian",
        backgrounds=("scout", "survivor", "raider"),
    )
    roller.queue(15, 3, 10)
    result = service.group_test(
        ["mara", "ulf"], "DEX", "slip past the watch together",
        "the way opens", "the cost lands", category="stealth",
    )
    assert result["ok"], result
    urchin_entry = next(e for e in result["individual"] if e["character_id"] == "mara")
    other_entry = next(e for e in result["individual"] if e["character_id"] == "ulf")
    assert urchin_entry["roll"]["edge"] == "advantage"
    assert other_entry["roll"]["edge"] == "single"


# -- usage dice --------------------------------------------------------------


def test_usage_roll_steps_the_resource_down(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(2)
    result = service.usage_roll("mara", "rations", "eat on the road")
    assert result["outcome"] == "downgraded"
    assert result["resource"]["current_die"] == "d4"
    assert service.store.read_character("mara").resources[0].die == "d4"


def test_a_depleted_resource_cannot_be_rolled(service: GameService, roller):
    make_fighter(service, roller)
    for value in (1, 1):
        roller.queue(value)
        service.usage_roll("mara", "rations", "eat on the road")
    result = service.usage_roll("mara", "rations", "eat on the road")
    assert result["ok"] is False
    assert result["error"] == "resource_depleted"


def test_unknown_resource_returns_a_structured_error(service: GameService, roller):
    make_fighter(service, roller)
    result = service.usage_roll("mara", "arrows", "loose a shaft")
    assert result["ok"] is False
    assert result["error"] == "resource_not_found"
