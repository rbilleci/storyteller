"""The prose is preserved verbatim; scope, reference, supersession and provenance live
beside it; ``visible_facts`` / ``hidden_facts`` survive as derived read-only
properties so every historical reader keeps its shape; the model addresses facts by
quotation, never by any identifier.
"""

from __future__ import annotations

import json

from conftest import ScriptedRoller, make_character

from bsh_mcp.models import Scene, Statement, live_statements
from bsh_mcp.service import GameService
from bsh_mcp.store import CampaignStore


def _last_event(service: GameService) -> dict:
    return service.store.read_events(limit=1)[-1]


def test_a_pre_migration_campaign_lifts_through_the_real_file_layer(service: GameService):
    """The objects-pin pattern: write, rewrite the raw file into the legacy shape,
    reload, and the record is intact with the derived properties unchanged."""
    service.scene_commit("The tide turns.", visible_changes=["A."], hidden_changes=["H."])
    raw = json.loads(service.store.state_path.read_text(encoding="utf-8"))
    assert "statements" in raw["scene"] and "visible_facts" not in raw["scene"]
    raw["scene"]["visible_facts"] = ["A."]
    raw["scene"]["hidden_facts"] = ["H."]
    del raw["scene"]["statements"]
    service.store.state_path.write_text(json.dumps(raw), encoding="utf-8")
    state = CampaignStore(service.store.root).read_state()
    assert state.scene.visible_facts == ["A."]
    assert state.scene.hidden_facts == ["H."]
    assert all(statement.seq == 0 for statement in state.scene.statements)
    # And the lifted record round-trips through a further commit untouched.
    service.scene_commit("Still turning.", visible_changes=["B."])
    state = service.store.read_state()
    assert state.scene.visible_facts == ["A.", "B."]


def test_statements_carry_seq_scope_and_source(service: GameService):
    result = service.scene_commit("The tide turns.", visible_changes=["A."], hidden_changes=["H."])
    sequence = int(result["event_id"].rsplit("-", 1)[1])
    state = service.store.read_state()
    by_text = {statement.text: statement for statement in state.scene.statements}
    assert by_text["A."].scope == "public" and by_text["A."].seq == sequence
    assert by_text["H."].scope == "world_hidden"
    assert by_text["A."].source == "model"


def test_exact_duplicates_are_skipped_per_scope_as_before(service: GameService):
    service.scene_commit("First.", visible_changes=["Rade is dead."])
    service.scene_commit("Again.", visible_changes=["Rade is dead. "], hidden_changes=["Rade is dead."])
    state = service.store.read_state()
    assert state.scene.visible_facts == ["Rade is dead."]
    assert state.scene.hidden_facts == ["Rade is dead."]  # a different scope is a different fact


def test_a_fact_retires_by_exact_or_normalized_quote_and_stays_in_history(service: GameService):
    service.scene_commit("Mud.", visible_changes=["Rill is sprawled in the mud on the road."])
    result = service.scene_commit(
        "Rill stands.",
        visible_changes=["Rill is on his feet, dripping mud."],
        retire_facts=["rill is sprawled in the mud on the road"],
    )
    sequence = int(result["event_id"].rsplit("-", 1)[1])
    state = service.store.read_state()
    assert state.scene.visible_facts == ["Rill is on his feet, dripping mud."]
    retired = next(s for s in state.scene.statements if "sprawled" in s.text)
    assert retired.superseded_at == sequence  # history, not deletion
    assert _last_event(service)["retired_facts"] == ["Rill is sprawled in the mud on the road."]
    rendered = service.store.scene_path.read_text(encoding="utf-8")
    assert "sprawled" not in rendered


def test_a_retired_text_may_be_re_established_and_a_bad_quote_only_warns(service: GameService):
    service.scene_commit("Barred.", visible_changes=["The tower door is barred."])
    service.scene_commit("Open.", retire_facts=["The tower door is barred."])
    service.scene_commit("Barred again.", visible_changes=["The tower door is barred."])
    state = service.store.read_state()
    assert state.scene.visible_facts == ["The tower door is barred."]
    assert sum("barred" in s.text.lower() for s in state.scene.statements) == 2
    result = service.scene_commit("Nothing.", retire_facts=["No such line."])
    assert result["ok"] is True
    assert any("matches no live fact" in warning for warning in result["warnings"])


def test_refs_validate_against_the_record_and_ride_every_statement(
    service: GameService, roller: ScriptedRoller
):
    make_character(service, roller, name="Rill")
    service.npc_create(name="Rade", level=1)
    result = service.scene_commit(
        "Rill faces Rade.",
        visible_changes=["Rill squares up to Rade."],
        refs=["rill", "rade", "ghost"],
    )
    assert any("names nothing recorded" in warning for warning in result["warnings"])
    state = service.store.read_state()
    fact = next(s for s in state.scene.statements if "squares up" in s.text)
    assert fact.refs == ("rill", "rade")
    assert _last_event(service)["refs"] == ["rill", "rade"]


def test_a_party_secret_is_party_knowledge_not_a_gm_secret(service: GameService):
    """B2, the arm-3 semantics typed: the party's own secret renders with the
    visible facts and never in the game-master-only section."""
    service.scene_commit(
        "The party hides the oar-case.",
        party_secrets=["The oar-case is hidden under the customs house floor."],
        hidden_changes=["Ammet watched them hide it."],
    )
    state = service.store.read_state()
    assert state.scene.visible_facts == ["The oar-case is hidden under the customs house floor."]
    assert state.scene.hidden_facts == ["Ammet watched them hide it."]
    rendered = service.store.scene_path.read_text(encoding="utf-8")
    visible_section = rendered.split("## Visible facts", 1)[1].split("## ", 1)[0]
    gm_section = rendered.split("## Game-master-only facts", 1)[1]
    assert "oar-case is hidden" in visible_section and "oar-case" not in gm_section
    assert "Ammet watched" in gm_section


def test_live_statements_is_the_one_derivation():
    statements = [
        Statement(seq=1, text="A.", scope="public"),
        Statement(seq=2, text="B.", scope="world_hidden"),
        Statement(seq=3, text="C.", scope="party_secret"),
        Statement(seq=4, text="D.", scope="public", superseded_at=9),
    ]
    assert [s.text for s in live_statements(statements)] == ["A.", "B.", "C."]
    assert [s.text for s in live_statements(statements, ("public", "party_secret"))] == ["A.", "C."]
    scene = Scene(statements=statements)
    assert scene.visible_facts == ["A.", "C."] and scene.hidden_facts == ["B."]
    assert "visible_facts" not in scene.model_dump()


def test_the_server_resolves_each_statements_own_refs_from_its_text(
    service: GameService, roller: ScriptedRoller
):
    """A voluntary commit usually omits the optional ``refs`` argument, so each
    statement's text is resolved against names the engine already knows -- per
    statement, not commit-wide, with the existing Unicode word-boundary matcher.
    """
    make_character(service, roller, name="Rill")
    service.npc_create(name="Rade", level=1)
    result = service.scene_commit(
        "Two things happen.",
        visible_changes=[
            "Rade drops his cleaver on the planks.",
            "The tide keeps falling.",
        ],
    )
    assert not result["warnings"]
    state = service.store.read_state()
    by_text = {s.text: s for s in state.scene.statements}
    assert by_text["Rade drops his cleaver on the planks."].refs == ("rade",)
    assert by_text["The tide keeps falling."].refs == ()


def test_text_resolved_refs_union_with_model_supplied_refs(
    service: GameService, roller: ScriptedRoller
):
    make_character(service, roller, name="Rill")
    service.npc_create(name="Rade", level=1)
    service.scene_commit(
        "Both hands.",
        visible_changes=["Rade eyes the coin pouch."],
        refs=["rill"],
    )
    state = service.store.read_state()
    fact = next(s for s in state.scene.statements if "coin pouch" in s.text)
    assert fact.refs == ("rill", "rade")


def test_text_resolution_respects_word_boundaries_and_object_slugs(
    service: GameService, roller: ScriptedRoller
):
    """"Radek" does not name Rade; a barrier's slug matches as words."""
    make_character(service, roller, name="Rill")
    service.npc_create(name="Rade", level=1)
    service.scene_commit(
        "Setup.",
        object_updates={"cellar-door": {"state": "barred"}},
    )
    service.scene_commit(
        "Two more.",
        visible_changes=[
            "Radek the trader poles past the dock.",
            "The cellar door holds against the tide.",
        ],
    )
    state = service.store.read_state()
    by_text = {s.text: s for s in state.scene.statements}
    assert by_text["Radek the trader poles past the dock."].refs == ()
    assert by_text["The cellar door holds against the tide."].refs == ("cellar-door",)
