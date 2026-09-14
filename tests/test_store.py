"""File-layer tests: validation, locking, atomic replacement, and the audit log."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import make_character
from filelock import FileLock

from bsh_mcp import store as store_module
from bsh_mcp.data import RulesData
from bsh_mcp.models import CampaignState
from bsh_mcp.service import GameService
from bsh_mcp.store import CampaignError, CampaignStore
from narrator.config import load_config


def test_initialise_creates_every_authoritative_file(campaign_root: Path):
    store = CampaignStore(campaign_root)
    store.initialize(title="The Ashen Bell")
    assert store.state_path.is_file()
    assert store.players_path.is_file()
    assert store.manifest_path.is_file()
    assert store.scene_path.is_file()
    assert store.events_path.is_file()
    assert store.read_manifest().title == "The Ashen Bell"


def test_initialise_refuses_to_overwrite_without_force(campaign_root: Path):
    store = CampaignStore(campaign_root)
    store.initialize()
    with pytest.raises(CampaignError) as error:
        store.initialize()
    assert error.value.code == "campaign_exists"


def test_missing_character_produces_a_structured_error(service: GameService):
    result = service.character_sheet("marta")
    assert result["ok"] is False
    assert result["error"] == "character_not_found"
    assert result["allowed_next_steps"]


@pytest.mark.parametrize(
    "identifier",
    ["../secret", "..", "a/b", "/etc/passwd", "Mara", "", "x" * 65, "mara\x00"],
)
def test_path_traversal_and_malformed_identifiers_are_rejected(
    service: GameService, identifier: str
):
    with pytest.raises(CampaignError) as error:
        service.store.character_path(identifier)
    assert error.value.code == "invalid_character_id"


def test_location_identifiers_are_confined_to_the_world_directory(service: GameService):
    with pytest.raises(CampaignError):
        service.store.location_path("../../etc/passwd")


def test_invalid_json_does_not_overwrite_a_valid_file(service: GameService, roller):
    make_character(service, roller)
    original = service.store.character_path("mara").read_text(encoding="utf-8")

    service.store.state_path.write_text("{ this is not json", encoding="utf-8")
    result = service.attribute_test("mara", "STR", "force the door", "the way opens", "the cost lands")
    assert result["ok"] is False
    assert result["error"] == "invalid_state_file"
    assert service.store.character_path("mara").read_text(encoding="utf-8") == original


def test_a_failed_transaction_writes_nothing(service: GameService, roller):
    make_character(service, roller)
    before_state = service.store.state_path.read_text(encoding="utf-8")
    before_events = service.store.events_path.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError):
        with service.store.transaction("test", actor_id="mara", reason="boom") as transaction:
            character = transaction.character("mara")
            character.hp = 1
            transaction.touch_character("mara")
            raise RuntimeError("deliberate failure")

    assert service.store.state_path.read_text(encoding="utf-8") == before_state
    assert service.store.events_path.read_text(encoding="utf-8") == before_events
    assert service.store.read_character("mara").hp != 1


def test_the_campaign_lock_blocks_a_second_writer(service: GameService, monkeypatch):
    monkeypatch.setattr(store_module, "LOCK_TIMEOUT_SECONDS", 0.05)
    holder = FileLock(str(service.store.lock_path))
    holder.acquire()
    try:
        with pytest.raises(CampaignError) as error:
            with service.store.transaction("test", reason="blocked"):
                pass
        assert error.value.code == "campaign_locked"
    finally:
        holder.release()


def test_no_temporary_files_survive_a_write(service: GameService, roller):
    make_character(service, roller)
    leftovers = [
        path
        for path in service.store.campaign_dir.rglob(".*.tmp-*")
        if path.is_file()
    ]
    assert leftovers == []


def test_a_late_failure_writes_nothing_not_even_the_earlier_dirty_character(
    service: GameService, roller, monkeypatch
):
    """The module's own documented invariant: 'If any step raises, the transaction
    writes nothing at all.' ``commit()`` used to write each dirty character file as
    it validated it, before state validation and scene rendering ran later in the
    same method -- so a failure in either of those left an already-written character
    file on disk with no matching state.json update and no audit event to explain
    it. Forcing the scene render (the last step before any write happens) to raise
    proves the fix: two passes, validate-and-render everything first, write only
    once nothing can raise.
    """
    make_character(service, roller, name="Mara")
    character_path = service.store.character_path("mara")
    before_character = character_path.read_text(encoding="utf-8")
    before_state = service.store.state_path.read_text(encoding="utf-8")
    before_event_count = len(service.store.read_events(limit=1000))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("scene render exploded")

    monkeypatch.setattr(service.store, "render_scene_markdown", _boom)

    with pytest.raises(RuntimeError, match="scene render exploded"):
        with service.store.transaction("probe", actor_id="mara", reason="force a late failure") as tx:
            character = tx.character("mara")
            character.coins += 5  # a real, otherwise-valid mutation
            tx.touch_character("mara")
            tx.scene_dirty = True  # forces the patched render to run during commit
            tx.record("a change that never lands")
            tx.commit({"outcome": "probe"})

    assert character_path.read_text(encoding="utf-8") == before_character
    assert service.store.state_path.read_text(encoding="utf-8") == before_state
    assert len(service.store.read_events(limit=1000)) == before_event_count


def test_every_mutation_increments_the_event_sequence(service: GameService, roller):
    make_character(service, roller)
    sequences = []
    for _ in range(3):
        roller.queue(9)
        result = service.attribute_test("mara", "STR", "shoulder the door", "the way opens", "the cost lands")
        assert result["ok"], result
        sequences.append(result["event_id"])

    assert sequences == ["evt-000002", "evt-000003", "evt-000004"]

    lines = [
        json.loads(line)
        for line in service.store.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["seq"] for event in lines] == [1, 2, 3, 4]
    assert service.store.read_state().event_seq == 4


def test_audit_events_record_tool_actor_and_roll(service: GameService, roller):
    make_character(service, roller)
    roller.queue(3)
    service.attribute_test("mara", "DEX", "cross the open mud", "the way opens", "the cost lands")
    events = service.store.read_events(limit=1)
    assert events[0]["tool"] == "attribute_test"
    assert events[0]["actor_id"] == "mara"
    assert events[0]["reason"] == "cross the open mud"
    assert events[0]["roll"]["selected"] == 3
    assert events[0]["outcome"] == "success"
    assert "prompt" not in events[0]


def test_a_fixed_clock_produces_deterministic_commit_timestamps(campaign_root: Path, roller):
    """One clock read per commit, reused across state, character, and event.

    Proves the actual value of the injected clock: a test can now assert an exact
    timestamp rather than merely that some string got set, and the same fixed
    instant lands identically in every file the commit touches.
    """
    fixed = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    game = GameService(campaign_root, roller=roller, clock=lambda: fixed)
    game.store.initialize(title="Clocked Campaign")
    make_character(game, roller)

    assert game.store.read_state().updated_at == "2026-03-04T05:06:07+00:00"
    assert game.store.read_character("mara").updated_at == "2026-03-04T05:06:07+00:00"
    events = game.store.read_events(limit=1)
    assert events[0]["at"] == "2026-03-04T05:06:07+00:00"


def test_state_survives_a_restart(service: GameService, roller):
    make_character(service, roller)
    roller.queue(2)
    service.attribute_test("mara", "STR", "haul the boat", "the way opens", "the cost lands")
    service.scene_commit("The party reaches the shrine wall.", in_game_time_delta_minutes=5)

    reopened = CampaignStore(service.store.root)
    state = reopened.read_state()
    assert state.in_game_minutes == 5
    assert state.scene.summary == "The party reaches the shrine wall."
    assert reopened.read_character("mara").name == "Mara"
    assert state.model_dump() == CampaignState.model_validate(
        json.loads(service.store.state_path.read_text(encoding="utf-8"))
    ).model_dump()


def test_scene_markdown_mirrors_state_and_marks_hidden_facts(service: GameService):
    service.scene_commit(
        "The bell tower stands silent.",
        visible_changes=["The tide has turned."],
        hidden_changes=["A listener watches from the customs house."],
        scene_title="Low water",
    )
    text = service.store.scene_path.read_text(encoding="utf-8")
    assert "Low water" in text
    assert "The tide has turned." in text
    assert "Game-master-only facts" in text
    assert "Never quote this section to players." in text


def test_scene_markdown_renders_the_typed_object_map(service: GameService):
    """A locked door's state now lives in one authoritative section instead of prose,
    so the render carries it exactly the same way it carries a clock's fill.
    """
    service.scene_commit(
        "The party finds the tower door barred.",
        object_updates={"tower-door": {"state": "locked", "note": "an iron slide bolt"}},
    )
    text = service.store.scene_path.read_text(encoding="utf-8")
    assert "## Objects" in text
    assert "- tower-door (locked) an iron slide bolt" in text
    assert [line for line in text.splitlines() if line != line.rstrip()] == []


def test_the_render_annotates_present_npcs_that_are_not_alive(service: GameService):
    """Test the render annotates present npcs that are not alive.
    """
    created = service.npc_create(name="Reed Thug", level=1, motive="rob")
    assert created["ok"], created
    state = service.store.read_state()
    assert "- reed-thug" in service.store.render_scene_markdown(state).splitlines()

    state.npcs["reed-thug"].status = "dead"
    lines = service.store.render_scene_markdown(state).splitlines()
    assert "- reed-thug (dead)" in lines
    assert "- reed-thug" not in lines

    state.npcs["reed-thug"].status = "fled"
    assert "- reed-thug (fled)" in service.store.render_scene_markdown(state).splitlines()


def test_scene_markdown_reports_no_objects_recorded_by_default(service: GameService):
    service.scene_commit("Nothing has changed hands yet.")
    text = service.store.scene_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    objects_index = lines.index("## Objects")
    assert lines[objects_index + 2] == "- None recorded."


def test_an_older_campaign_without_the_objects_field_validates_unchanged(
    service: GameService,
):
    """Test an older campaign without the objects field validates unchanged.
    """
    service.scene_commit(
        "The party reaches the shrine wall.",
        visible_changes=["The tide has turned."],
        exits=["the-road-shrine"],
    )
    raw = json.loads(service.store.state_path.read_text(encoding="utf-8"))
    assert "objects" in raw["scene"]  # the file this service just wrote does carry it
    del raw["scene"]["objects"]  # simulate a campaign written before this field existed
    service.store.state_path.write_text(json.dumps(raw), encoding="utf-8")

    reopened = CampaignStore(service.store.root)
    state = reopened.read_state()
    assert state.scene.objects == {}
    assert state.scene.visible_facts == ["The tide has turned."]
    assert state.scene.exits == ["the-road-shrine"]

    # The reloaded state re-renders and re-serializes cleanly, exactly like any other
    # campaign this service manages -- the missing field is not a permanent scar.
    text = reopened.render_scene_markdown(state)
    assert "## Objects" in text
    assert "- None recorded." in text


def test_the_default_data_loader_is_process_cached(campaign_root: Path):
    """Names the tradeoff a caller opts out of by injecting a loader: the default
    is fast (one parse per rules_dir per process) but blind to a same-session
    rules/*.json edit, since the two reads below return the identical object."""
    game = GameService(campaign_root)
    first, second = game.data, game.data
    assert first is second


def test_an_injected_data_loader_bypasses_the_process_cache(campaign_root: Path):
    """A caller that needs to see a rules/*.json edit without restarting --
    a test, a hot-reload tool -- injects an uncached loader instead."""
    calls: list[str] = []

    def uncached(rules_dir: str) -> RulesData:
        calls.append(rules_dir)
        return RulesData.load(Path(rules_dir))

    game = GameService(campaign_root, data_loader=uncached)
    first, second = game.data, game.data
    assert first is not second
    assert len(calls) == 2


def test_backup_creates_an_archive(service: GameService):
    archive = service.store.backup(label="test")
    assert archive.is_file()
    assert archive.suffix == ".gz"
    assert archive.parent == service.store.backups_dir


def test_a_fixed_clock_produces_a_deterministic_backup_stamp(campaign_root: Path):
    store = CampaignStore(campaign_root, clock=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    store.initialize(title="Clocked Campaign")
    archive = store.backup(label="test")
    assert archive.name == "campaign-20260102-030405-test.tar.gz"


def test_character_file_is_valid_json_after_every_write(service: GameService, roller):
    make_character(service, roller)
    roller.queue(19)
    service.attribute_test("mara", "STR", "lift the portcullis", "the way opens", "the cost lands")
    payload = json.loads(service.store.character_path("mara").read_text(encoding="utf-8"))
    assert payload["id"] == "mara"
    assert payload["hp"] == payload["hp_max"]


def test_players_file_links_discord_ids_to_characters(service: GameService, roller):
    make_character(service, roller, name="Mara")
    make_character(service, roller, name="Ulf", origin="civilised", backgrounds=("bodyguard", "legionnaire", "scout"))
    players = service.store.read_players()
    assert {link.character_id for link in players.players} == {"mara", "ulf"}
    assert players.by_discord_id("discord-mara").character_id == "mara"
    assert players.by_character_id("ulf").display_name == "Ulf"


def test_rules_tables_load_from_the_repository(service: GameService):
    data = service.data
    assert data.starting_coins("decadent") == 100
    assert "brutal" in data.implemented_effects()
    assert data.helpless_result(6)["effect"] == "death"


def test_resolve_character_id_precedence(service: GameService, roller):
    make_character(service, roller, name="Mara")
    make_character(service, roller, name="Ulf")
    store = service.store
    assert store.resolve_character_id("mara") == "mara"       # exact id
    assert store.resolve_character_id("MARA") == "mara"       # case-folded id
    assert store.resolve_character_id("Ulf") == "ulf"         # sheet name
    with pytest.raises(CampaignError) as error:
        store.resolve_character_id("Bogus")
    assert error.value.code == "character_not_found"
    assert any("mara" in step for step in error.value.allowed_next_steps)


def test_resolution_never_touches_the_path_layer_with_raw_input(service: GameService):
    """A traversal string fails resolution before any path is built from it."""
    with pytest.raises(CampaignError) as error:
        service.store.resolve_character_id("../secret")
    assert error.value.code == "character_not_found"


def test_resolve_character_id_reports_ambiguous_names(service: GameService, roller):
    """Two sheets can share a display name (only ids are unique); resolving that
    name must refuse rather than silently pick one. The name has to be two words
    -- a single word like "Rade" slugifies to an id ("rade") that case-folds to
    the same string as the name itself, so the first sheet's own id would match
    before resolution ever reaches the name fallback this test means to exercise.
    """
    make_character(service, roller, name="Reed Thug")
    make_character(service, roller, name="Reed Thug")  # gets id "reed-thug-2"
    with pytest.raises(CampaignError) as error:
        service.store.resolve_character_id("Reed Thug")
    assert error.value.code == "character_ambiguous"
    assert "reed-thug" in error.value.message and "reed-thug-2" in error.value.message


def test_resolve_npc_id_precedence_and_ambiguity(service: GameService, roller):
    """``_resolve_npc_id`` shares its precedence with ``resolve_character_id`` via
    the same ``resolve_or_raise`` helper; this pins it against the NPC collection
    directly, including the ambiguous-name branch nothing exercised before.
    """
    make_character(service, roller)
    npc_a = service.npc_create(name="Reed Thug", level=1, motive="rob")
    assert npc_a["ok"], npc_a
    npc_b = service.npc_create(name="Reed Thug", level=1, motive="rob")  # id "reed-thug-2"
    assert npc_b["ok"], npc_b

    with service.store.transaction("probe", reason="resolve") as tx:
        assert service._resolve_npc_id(tx, "reed-thug") == "reed-thug"
        assert service._resolve_npc_id(tx, "REED-THUG") == "reed-thug"
        with pytest.raises(CampaignError) as ambiguous:
            service._resolve_npc_id(tx, "Reed Thug")
        assert ambiguous.value.code == "npc_ambiguous"
        with pytest.raises(CampaignError) as not_found:
            service._resolve_npc_id(tx, "nobody")
        assert not_found.value.code == "npc_not_found"
        tx.commit({"outcome": "probe"})


def test_resolve_npc_id_by_name_warns_with_the_resolved_id(service: GameService, roller):
    make_character(service, roller)
    created = service.npc_create(name="Reed Thug", level=1, motive="rob")
    assert created["ok"], created
    with service.store.transaction("probe", reason="resolve") as tx:
        resolved = service._resolve_npc_id(tx, "Reed Thug".upper())
        assert resolved == "reed-thug"
        assert any("reed-thug" in warning for warning in tx.warnings)
        tx.commit({"outcome": "probe"})


def test_resolve_combat_actor_precedence_and_ambiguity(service: GameService, roller):
    """``_resolve_combat_actor``'s error shape is deliberately its own (combat
    order, not a full id list; ``actor_not_in_combat``, not ``*_not_found``), so
    this pins it directly rather than through ``resolve_or_raise``.
    """
    make_character(service, roller, name="Mara")
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob")
    assert npc["ok"], npc
    started = service.combat_start(
        pc_ids=["mara"], npc_ids=[npc["npc_id"]],
        initial_ranges={npc["npc_id"]: "close"}, reason="ambush",
    )
    assert started["ok"], started

    with service.store.transaction("probe", reason="resolve") as tx:
        combat = tx.state.combat
        assert service._resolve_combat_actor(tx, combat, "mara") == "mara"
        assert service._resolve_combat_actor(tx, combat, "MARA") == "mara"
        assert service._resolve_combat_actor(tx, combat, "Mara") == "mara"  # by name
        with pytest.raises(CampaignError) as not_in_combat:
            service._resolve_combat_actor(tx, combat, "nobody")
        assert not_in_combat.value.code == "actor_not_in_combat"
        assert "Combat order" in not_in_combat.value.allowed_next_steps[0]
        tx.commit({"outcome": "probe"})


def test_resolve_combat_actor_reports_ambiguous_names(service: GameService, roller):
    """Two NPCs sharing a name, both in the same fight, must refuse a name lookup
    rather than pick one -- the same ambiguity guarantee as the character and NPC
    resolvers, pinned here against ``_resolve_combat_actor``'s own error shape.
    """
    make_character(service, roller, name="Mara")
    first = service.npc_create(name="Reed Thug", level=1, motive="rob")
    second = service.npc_create(name="Reed Thug", level=1, motive="rob")
    assert first["ok"] and second["ok"]
    started = service.combat_start(
        pc_ids=["mara"],
        npc_ids=[first["npc_id"], second["npc_id"]],
        initial_ranges={first["npc_id"]: "close", second["npc_id"]: "close"},
        reason="ambush",
    )
    assert started["ok"], started

    with service.store.transaction("probe", reason="resolve") as tx:
        with pytest.raises(CampaignError) as ambiguous:
            service._resolve_combat_actor(tx, tx.state.combat, "Reed Thug")
        assert ambiguous.value.code == "actor_ambiguous"
        assert first["npc_id"] in ambiguous.value.message
        assert second["npc_id"] in ambiguous.value.message
        tx.commit({"outcome": "probe"})


# -- the rendered combat section ---------------------------------------------


def _open_a_fight(service: GameService, roller) -> str:
    """Create one fighter and one opponent, then roll initiative."""
    make_character(service, roller, name="Mara")
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
    return npc["npc_id"]


def test_the_scene_render_omits_combat_while_no_fight_runs(service: GameService, roller):
    """A campaign outside combat renders the bytes it rendered before the section existed."""
    make_character(service, roller, name="Mara")
    assert "## Combat" not in service.store.scene_path.read_text(encoding="utf-8")


def test_a_combat_mutating_commit_re_renders_the_scene(service: GameService, roller):
    """``combat_start`` sets no ``scene_dirty``, so only the combat comparison refreshes this."""
    npc_id = _open_a_fight(service, roller)
    scene = service.store.scene_path.read_text(encoding="utf-8")
    assert "## Combat" in scene
    assert "- round: 1" in scene
    assert f"- combatant: {npc_id} side=npc range=close" in scene
    # The renderer omits the band for a player character, because ``combat_start``
    # records one for each opponent and none for a player character.
    assert "- combatant: mara side=pc actions=" in scene
    assert "- combatant: mara side=pc range=" not in scene
    order_line = next(line for line in scene.splitlines() if line.startswith("- order:"))
    assert "mara" in order_line and npc_id in order_line


def test_the_render_drops_the_combat_section_when_the_fight_ends(
    service: GameService, roller
):
    """Closing a fight must refresh the render, not leave the last open round resident."""
    _open_a_fight(service, roller)
    assert "## Combat" in service.store.scene_path.read_text(encoding="utf-8")
    with service.store.transaction("probe", actor_id="mara", reason="end the fight") as tx:
        tx.state.combat.active = False
        tx.record("the fight ended")
        tx.commit({"outcome": "probe"})
    assert "## Combat" not in service.store.scene_path.read_text(encoding="utf-8")


def test_a_commit_that_leaves_combat_alone_does_not_re_render(service: GameService, roller):
    """The comparison must refresh on a combat change only, never on every commit."""
    _open_a_fight(service, roller)
    before = service.store.scene_path.read_text(encoding="utf-8")
    stamp = service.store.scene_path.stat().st_mtime_ns
    with service.store.transaction("probe", actor_id="mara", reason="touch nothing") as tx:
        tx.record("an unrelated change")
        tx.commit({"outcome": "probe"})
    assert service.store.scene_path.read_text(encoding="utf-8") == before
    assert service.store.scene_path.stat().st_mtime_ns == stamp


def test_the_render_tracks_the_round_a_fight_advances_to(service: GameService, roller):
    """Test the render tracks the round a fight advances to.
    """
    _open_a_fight(service, roller)
    with service.store.transaction("probe", actor_id="mara", reason="advance") as tx:
        tx.state.combat.round = 4
        tx.record("the fight advanced")
        tx.commit({"outcome": "probe"})
    assert "- round: 4" in service.store.scene_path.read_text(encoding="utf-8")


def test_render_scene_markdowns_combat_section_round_trips_through_its_reader(
    service: GameService, roller
):
    """``render_scene_markdown``'s combat section is a parsed contract for
    ``narrator.interactions.read_combat_snapshot`` (both docstrings say so), but until
    now nothing checked the two agree: each side was tested only against its own
    separately hand-maintained fixture text, so a real format drift between the writer
    and the reader could pass every existing test on both sides of the boundary.
    """
    from narrator.interactions import read_combat_snapshot

    npc_id = _open_a_fight(service, roller)
    state = service.store.read_state()
    combat = state.combat
    assert combat.active is True  # otherwise this test would trivially pass

    snapshot = read_combat_snapshot(service.store.root)

    assert snapshot.active == combat.active
    assert snapshot.round == combat.round
    assert snapshot.active_actor == combat.active_actor
    assert snapshot.order == tuple(combat.order)
    assert snapshot.side_of("mara") == combat.actors["mara"].side
    assert snapshot.side_of(npc_id) == combat.actors[npc_id].side
    assert set(snapshot.npc_combatants) == {
        actor_id for actor_id, actor in combat.actors.items() if actor.side == "npc"
    }


# -- root-discovery parity between the two interpreters ----------------------
#
# ``bsh_mcp.store.discover_root`` and ``narrator.config.load_config`` each
# implement the same explicit-argument > BSH_CAMPAIGN_ROOT > repository-root
# precedence independently, because the two live in separate interpreters (see
# CONTRIBUTING.md, "Two processes, by necessity") and cannot share the implementation
# directly. ``narrator.config`` is stdlib-only, so it imports cleanly here in the
# project environment even though it is never run under it; these tests catch the
# two copies drifting apart, which nothing else does.


def test_root_discovery_agrees_on_an_explicit_argument(tmp_path: Path):
    store_root = store_module.discover_root(str(tmp_path))
    narrator_root = load_config(str(tmp_path)).campaign_root
    assert store_root == narrator_root == tmp_path.expanduser().resolve()


def test_root_discovery_agrees_on_the_environment_variable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BSH_CAMPAIGN_ROOT", str(tmp_path))
    store_root = store_module.discover_root()
    narrator_root = load_config().campaign_root
    assert store_root == narrator_root == tmp_path.expanduser().resolve()


def test_root_discovery_agrees_on_the_repository_fallback(monkeypatch):
    monkeypatch.delenv("BSH_CAMPAIGN_ROOT", raising=False)
    assert store_module.discover_root() == load_config().campaign_root
