"""Fiction-debt ledger and stakes tool tests."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from conftest import make_character
from tools_shared import make_fighter

from bsh_mcp.models import EnginePurchaseRequest, MerchantStock
from bsh_mcp.server import create_server
from bsh_mcp.service import GameService
from bsh_mcp.store import CampaignStore

# -- fiction-debt ledger and stakes (durable-fiction Phase 1) -----------------


def test_attribute_test_requires_nonempty_stakes(service: GameService, roller):
    """A roll with no declared stakes should not be rolled, and nothing is written."""
    make_fighter(service, roller)
    before = service.store.read_state().event_seq
    for success, failure in (("", "the cost lands"), ("the way opens", "   "), ("", "")):
        result = service.attribute_test("mara", "STR", "force the door", success, failure)
        assert result["ok"] is False
        assert result["error"] == "empty_stakes"
    assert service.store.read_state().event_seq == before


def test_identical_or_reason_echo_stakes_warn_but_pass(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5)
    result = service.attribute_test(
        "mara", "STR", "force the door", "the guards hear it", "the guards hear it"
    )
    assert result["ok"], result
    assert any("identical" in warning for warning in result["warnings"])
    roller.queue(5)
    result = service.attribute_test(
        "mara", "STR", "force the door", "force the door", "the hinge holds"
    )
    assert result["ok"], result
    assert any("repeats the reason" in warning for warning in result["warnings"])


def test_realized_branch_maps_criticals_to_base_branches(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(1)
    natural_one = service.attribute_test(
        "mara", "STR", "lift the gate", "the gate rises", "the gate jams"
    )
    assert natural_one["outcome"] == "critical_success"
    assert natural_one["realized"] == "success"
    assert "the gate rises" in natural_one["narration_facts"]

    roller.queue(20, 3)  # natural 20 plus the mandatory Doom roll
    natural_twenty = service.attribute_test(
        "mara", "STR", "lift the gate", "the gate rises", "the gate jams"
    )
    assert natural_twenty["outcome"] == "critical_failure"
    assert natural_twenty["realized"] == "failure"
    assert "the gate jams" in natural_twenty["narration_facts"]


def test_hidden_stakes_never_reach_narration_facts(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5)
    result = service.attribute_test(
        "mara", "DEX", "cross the mud", "she crosses unseen", "she is seen",
        "the listener marks her either way",
    )
    assert result["ok"], result
    assert result["stakes"]["hidden"] == "the listener marks her either way"
    assert all("listener" not in fact for fact in result["narration_facts"])


def test_stakes_roll_creates_debt_and_scene_commit_ratifies_it(
    service: GameService, roller
):
    make_fighter(service, roller)
    roller.queue(5)
    rolled = service.attribute_test(
        "mara", "DEX", "cross the mud", "she reaches the wall", "the patrol turns"
    )
    state = service.store.read_state()
    assert len(state.fiction_debt) == 1
    debt = state.fiction_debt[0]
    assert debt.tool == "attribute_test"
    assert debt.seq == int(rolled["event_id"].split("-")[1])
    assert debt.realized == "success"
    assert debt.stakes.success == "she reaches the wall"

    committed = service.scene_commit("Mara reaches the shrine wall.")
    assert committed["ok"], committed
    state = service.store.read_state()
    assert state.fiction_debt == []
    events = service.store.read_events(limit=1)
    assert events[0]["ratified_seqs"] == [debt.seq]


def test_debt_survives_a_restart(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5)
    service.attribute_test(
        "mara", "DEX", "cross the mud", "she reaches the wall", "the patrol turns"
    )
    reopened = CampaignStore(service.store.root)
    debt = reopened.read_state().fiction_debt
    assert len(debt) == 1
    assert debt[0].stakes.failure == "the patrol turns"


def test_unratified_outcomes_reach_the_on_disk_scene_file(
    service: GameService, roller
):
    """The dominant failure flow is a roll with no scene_commit. The on-disk
    scene.md must show the unratified outcome in that exact flow, with hidden
    stakes confined to the game-master-only half."""
    make_fighter(service, roller)
    roller.queue(5)
    service.attribute_test(
        "mara", "DEX", "cross the mud", "she reaches the wall", "the patrol turns",
        "the listener marks her",
    )
    text = service.store.scene_path.read_text(encoding="utf-8")
    assert "## Unratified outcomes" in text
    public_section = text.split("## Game-master-only facts")[0]
    assert "she reaches the wall" in public_section
    assert "the listener marks her" not in public_section
    assert "### Unratified GM-only outcomes" in text
    assert "the listener marks her" in text.split("## Game-master-only facts")[1]

    service.scene_commit("Mara reaches the wall; the scene absorbs it.")
    ratified = service.store.scene_path.read_text(encoding="utf-8")
    unratified_section = ratified.split("## Unratified outcomes")[1].split("##")[0]
    assert "she reaches the wall" not in unratified_section
    assert "- None recorded." in unratified_section


def test_group_test_records_group_stakes(service: GameService, roller):
    make_fighter(service, roller)
    make_character(
        service, roller, name="Ulf", origin="civilised",
        backgrounds=("bodyguard", "legionnaire", "sword-master"),
        weapons=("arming sword",),
    )
    roller.queue(4, 18)
    result = service.group_test(
        ["mara", "ulf"], "DEX", "slip past the guard",
        "the group slips through", "the guard raises the alarm",
    )
    assert result["ok"], result
    assert result["realized"] == "success"
    assert "the group slips through" in result["narration_facts"]
    debt = service.store.read_state().fiction_debt
    assert debt[-1].tool == "group_test"
    assert debt[-1].realized == "success"


def test_usage_depletion_enters_the_ledger_without_stakes(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(2)
    service.usage_roll("mara", "rations", "eat on the road")
    debt = service.store.read_state().fiction_debt
    assert debt[-1].tool == "usage_roll"
    assert debt[-1].outcome == "downgraded"
    assert debt[-1].realized == "failure"


def test_combat_stakes_declared_at_start_realize_at_the_end(
    service: GameService, roller
):
    make_fighter(service, roller)
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob")
    roller.queue(5)
    started = service.combat_start(
        ["mara"], [npc["npc_id"]], {npc["npc_id"]: "close"},
        reason="ambush on the planks",
        stakes_success="the ambush is broken and the planks hold",
        stakes_failure="the party is driven into the mud",
    )
    assert started["ok"], started
    service.combat_begin_turn("mara")
    roller.queue(5, 6)
    # The killing blow ends the fight itself, so the stakes crystallize on the
    # attack that ended it rather than on a separate combat_end_turn call.
    ended = service.combat_attack("mara", npc["npc_id"])
    assert ended["combat_over"] is True
    debt = service.store.read_state().fiction_debt
    combat_entries = [d for d in debt if d.tool == "combat_attack" and d.outcome == "combat_ended"]
    assert len(combat_entries) == 1
    assert combat_entries[0].realized == "success"
    assert combat_entries[0].stakes.success == "the ambush is broken and the planks hold"


def test_rest_helpless_and_npc_create_enter_the_ledger(service: GameService, roller):
    make_fighter(service, roller)
    service.npc_create(name="Reed Thug", level=1, motive="rob the party")
    service.rest(["mara"], "short", reason="catch breath")
    tools = [d.tool for d in service.store.read_state().fiction_debt]
    assert "npc_create" in tools
    assert "rest" in tools


def test_npc_create_debt_carries_its_own_stake_not_the_npcs_motive(
    service: GameService, roller
):
    """Test npc create debt carries its own stake not the npcs motive.
    """
    make_fighter(service, roller)
    service.npc_create(name="Rade", level=1, motive="earn enough coins to pay the market tax")
    debt = service.store.read_state().fiction_debt
    entry = next(d for d in debt if d.tool == "npc_create")
    assert entry.reason == "create Rade"
    assert entry.realized == "success"
    assert entry.stakes.success == "Rade is present at the scene (earn enough coins to pay the market tax)"
    assert entry.realized_public_text() == entry.stakes.success


def test_npc_create_combat_start_and_begin_turn_leave_settleable_debt(
    service: GameService, roller
):
    """Test npc create combat start and begin turn leave settleable debt.
    """
    make_fighter(service, roller)
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob the party")
    assert npc["ok"], npc
    roller.queue(5)
    started = service.combat_start(
        pc_ids=["mara"],
        npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: "close"},
        reason="ambush on the plank walk",
    )
    assert started["ok"], started
    opened = service.combat_begin_turn("mara")
    assert opened["ok"], opened

    debt = service.store.read_state().fiction_debt
    assert [entry.tool for entry in debt] == ["npc_create"], (
        "the reproduction isolates npc_create as the sole source of debt in this "
        "sequence -- combat_start and combat_begin_turn must add none of their own"
    )

    settled = service.ledger_settle(reason="the setup stands; the attack roll never happened")
    assert settled["ok"], settled
    assert service.store.read_state().fiction_debt == []


def test_campaign_status_reports_uncommitted_fiction_and_recent_events(
    service: GameService, roller
):
    make_fighter(service, roller)
    roller.queue(5)
    service.attribute_test(
        "mara", "DEX", "cross the mud", "she reaches the wall", "the patrol turns"
    )
    status = service.campaign_status()
    assert status["ok"], status
    assert len(status["uncommitted_fiction"]) == 1
    entry = status["uncommitted_fiction"][0]
    assert entry["stakes"]["success"] == "she reaches the wall"
    assert entry["realized"] == "success"
    recent = status["recent_events"]
    assert recent[-1]["tool"] == "attribute_test"
    assert recent[-1]["reason"] == "cross the mud"


def test_scene_commit_event_records_the_fiction_it_commits(service: GameService):
    result = service.scene_commit(
        "The tide turns.",
        visible_changes=["The plank walk floods."],
        hidden_changes=["A second bell has cracked."],
        in_game_time_delta_minutes=30,
        new_clocks=[{"id": "tide", "name": "The turning tide", "segments": 6}],
    )
    assert result["ok"], result
    event = service.store.read_events(limit=1)[0]
    assert event["visible_changes"] == ["The plank walk floods."]
    assert event["hidden_changes"] == ["A second bell has cracked."]
    assert event["in_game_time_delta_minutes"] == 30
    assert event["new_clocks"][0]["id"] == "tide"


def test_scene_commit_event_records_applied_changes_not_submitted_arguments(
    service: GameService,
):
    """A duplicate clock and an unmatched hook are skipped with warnings; the
    audit event must not claim they happened."""
    service.scene_commit(
        "Setup.", new_clocks=[{"id": "tide", "name": "The tide", "segments": 6}],
        new_hooks=["Find the bell-keeper."],
    )
    result = service.scene_commit(
        "The duplicate and the ghost.",
        new_clocks=[{"id": "tide", "name": "The tide again", "segments": 4}],
        resolved_hooks=["A hook nobody recorded."],
        new_hooks=["Find the bell-keeper."],
    )
    assert result["ok"], result
    event = service.store.read_events(limit=1)[0]
    assert event["new_clocks"] == []
    assert event["resolved_hooks"] == []
    assert event["new_hooks"] == []
    assert len(result["warnings"]) >= 2


def test_engine_confirmed_purchase_mutates_coins_and_equipment_with_an_audit_event(
    service: GameService, roller,
):
    """Test engine confirmed purchase mutates coins and equipment with an audit event.
    """
    created = make_fighter(service, roller, name="Ossa")
    buyer_id = created["character_id"]
    with service.store.transaction("test", buyer_id, "fund buyer") as transaction:
        buyer = transaction.character(buyer_id)
        starting_coins = buyer.coins
        transaction.state.merchant_stocks["quay-merchant"] = MerchantStock(
            version=1, items={"lantern": 3}, prices={"lantern": 4}
        )
        transaction.commit({"outcome": "setup"})

    def _digest(label: str) -> str:
        return sha256(label.encode("utf-8")).hexdigest()

    request = EnginePurchaseRequest(
        decision_key="decision-key-lantern-001",
        action_fingerprint=_digest("action"),
        trade_fingerprint=_digest("trade"),
        idempotency_key=_digest("receipt"),
        buyer_id=buyer_id,
        seller_id="quay-merchant",
        item_id="lantern",
        quantity=1,
        price_copper=4,
        stock_version=1,
    )
    # A narrator-adjacent, un-capabilitied call must refuse -- the tool surface a
    # model reaches never includes this method, and this is the mechanical proof.
    direct = service.engine_confirm_purchase(request)
    assert direct["ok"] is False and direct["error"] == "purchase_capability_required"

    events_before = len(service.store.read_events(100))
    confirmed = service.narrator_purchase_executor().confirm(request)
    assert confirmed["ok"] is True and confirmed["outcome"] == "confirmed"

    after = service.store.read_character(buyer_id)
    assert after.coins == starting_coins - 4
    assert after.equipment == ["lantern"]
    stock = service.store.read_state().merchant_stocks["quay-merchant"]
    assert stock.items["lantern"] == 2 and stock.version == 2

    events_after = service.store.read_events(100)
    assert len(events_after) == events_before + 1
    event = events_after[-1]
    assert event["tool"] == "engine_purchase"
    assert event["actor_id"] == buyer_id
    assert event["outcome"] == "confirmed_purchase"
    assert event["purchase"] == "confirmed"
    assert "confirmed engine purchase" in event["changes"]


async def test_protocol_schema_requires_stakes(campaign_root: Path):
    server = create_server(campaign_root, seed=1)
    for tool in await server.list_tools():
        if tool.name in ("attribute_test", "group_test"):
            required = set((tool.input_schema or {}).get("required", []))
            assert "stakes_success" in required, tool.name
            assert "stakes_failure" in required, tool.name
            assert "stakes_hidden" not in required, tool.name


def test_ledger_settle_clears_the_debt_and_records_the_waived_stakes(service, roller):
    make_fighter(service, roller)
    roller.queue(20)
    service.attribute_test(
        "mara", "WIS", "listen at the crypt door",
        "Mara hears the keeper below.", "Mara hears nothing and stays unaware.",
    )
    state = json.loads((service.store.root / "campaign" / "state.json").read_text())
    assert state["fiction_debt"], "the roll must open debt for this test to mean anything"

    result = service.ledger_settle(reason="The failed listen changed nothing durable.")

    assert result["ok"] is True
    assert result["outcome"] == "ledger_waived"
    state = json.loads((service.store.root / "campaign" / "state.json").read_text())
    assert state["fiction_debt"] == []
    events = [
        json.loads(line)
        for line in (service.store.root / "campaign" / "logs" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    settle_events = [event for event in events if event["tool"] == "ledger_settle"]
    assert len(settle_events) == 1
    # RS-WAIVE-AUDITED demands content, not existence: the event must preserve the
    # stated reason and every waived stake. The roll above declared stakes and rolled
    # a scripted 20 against WIS 13, so the failure stake is the realized text.
    event = settle_events[0]
    assert event["outcome"] == "ledger_waived"
    assert event["reason"] == "The failed listen changed nothing durable."
    assert len(event["waived"]) == 1
    waived = event["waived"][0]
    assert waived["tool"] == "attribute_test"
    assert waived["realized_public_text"] == "Mara hears nothing and stays unaware."


def test_a_waived_stake_stays_tool_visible_through_campaign_status(service, roller):
    """An audit found waived stakes surviving only in the raw log: campaign_status's
    projection dropped the waived array, so a stake-bearing waive left every surface
    a resuming session reads. The projection must carry it."""
    make_fighter(service, roller)
    roller.queue(20)
    service.attribute_test(
        "mara", "WIS", "listen at the crypt door",
        "Mara hears the keeper below.", "Mara hears nothing and stays unaware.",
    )
    service.ledger_settle(reason="Covered by the narration.")

    status = service.campaign_status()
    settle_rows = [
        event for event in status["recent_events"] if event["tool"] == "ledger_settle"
    ]
    assert len(settle_rows) == 1
    waived = settle_rows[0]["waived"]
    assert waived[0]["tool"] == "attribute_test"
    assert waived[0]["realized_public_text"] == "Mara hears nothing and stays unaware."
    # Non-settle events must not grow a waived key.
    assert all("waived" not in event for event in status["recent_events"] if event["tool"] != "ledger_settle")


def test_ledger_settle_rejects_a_blank_reason(service, roller):
    make_fighter(service, roller)
    roller.queue(20)
    service.attribute_test("mara", "WIS", "listen", "hears", "misses")

    result = service.ledger_settle(reason="   ")

    assert result["ok"] is False
    assert result["error"] == "empty_waive_reason"
    state = json.loads((service.store.root / "campaign" / "state.json").read_text())
    assert state["fiction_debt"], "a rejected settle must leave the debt standing"


def test_ledger_settle_rejects_an_already_clear_ledger(service):
    result = service.ledger_settle(reason="nothing outstanding")
    assert result["ok"] is False
    assert result["error"] == "ledger_already_clear"


# -- session_close ledger gate (durable-fiction Phase 2) ----------------------


def test_session_close_refuses_with_open_debt(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5)
    rolled = service.attribute_test(
        "mara", "DEX", "cross the mud", "she reaches the wall", "the patrol turns"
    )
    seq = int(rolled["event_id"].split("-")[1])

    result = service.session_close("Night falls", "The party makes camp.")

    assert result["ok"] is False
    assert result["error"] == "uncommitted_fiction_debt"
    assert f"[{seq}]" in result["message"]
    assert "she reaches the wall" in result["message"]
    state = service.store.read_state()
    assert len(state.fiction_debt) == 1, "a refused close must leave the debt standing"
    assert state.session == 1, "a refused close must not open the next session"


def test_session_close_proceeds_once_the_debt_is_ratified(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5)
    service.attribute_test(
        "mara", "DEX", "cross the mud", "she reaches the wall", "the patrol turns"
    )
    service.scene_commit("Mara reaches the shrine wall.")

    result = service.session_close("Night falls", "The party makes camp.")

    assert result["ok"], result
    assert result["accepted_uncommitted_debt"] == []


def test_session_close_accepts_uncommitted_debt_when_told_to(service: GameService, roller):
    make_fighter(service, roller)
    roller.queue(5)
    rolled = service.attribute_test(
        "mara", "DEX", "cross the mud", "she reaches the wall", "the patrol turns"
    )
    seq = int(rolled["event_id"].split("-")[1])

    result = service.session_close(
        "Night falls", "The party makes camp.", accept_uncommitted=True
    )

    assert result["ok"], result
    assert result["accepted_uncommitted_debt"] == [
        {
            "seq": seq,
            "tool": "attribute_test",
            "outcome": "success",
            "realized_public_text": "she reaches the wall",
        }
    ]
    # The override records the decision but does not waive the debt -- only
    # scene_commit/ledger_settle ever clear fiction_debt.
    state = service.store.read_state()
    assert len(state.fiction_debt) == 1
    event = [e for e in service.store.read_events(limit=5) if e["tool"] == "session_close"][-1]
    assert event["accepted_uncommitted_debt"] == result["accepted_uncommitted_debt"]


def test_session_close_with_no_debt_reports_empty_acceptance_list(service: GameService):
    result = service.session_close("Night falls", "The party makes camp.")
    assert result["ok"], result
    assert result["accepted_uncommitted_debt"] == []


def test_session_close_summary_records_accepted_uncommitted_debt(
    service: GameService, roller
):
    """The audit event alone is not what a future session reads -- the summary
    file is. An accepted close must not let the realized stake vanish from the
    one record session_zero-style recaps are built from."""
    make_fighter(service, roller)
    roller.queue(5)
    service.attribute_test(
        "mara", "DEX", "cross the mud", "she reaches the wall", "the patrol turns"
    )

    result = service.session_close(
        "Night falls", "The party makes camp.", accept_uncommitted=True
    )
    assert result["ok"], result

    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "## Unresolved outcomes" in summary
    section = summary.split("## Unresolved outcomes")[1].split("##")[0]
    assert "she reaches the wall" in section


def test_session_close_summary_omits_the_section_with_no_accepted_debt(
    service: GameService,
):
    result = service.session_close("Night falls", "The party makes camp.")
    assert result["ok"], result
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "## Unresolved outcomes" not in summary
