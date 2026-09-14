"""scene_commit tool tests.
"""

from __future__ import annotations

from pathlib import Path

from conftest import make_character
from tools_shared import drop_to_helpless, make_fighter, open_fight

from bsh_mcp.models import traversal_clock_id
from bsh_mcp.service import GameService

# -- scene and session -------------------------------------------------------


def test_scene_commit_updates_state_and_clocks(service: GameService, roller):
    result = service.scene_commit(
        "The party reaches the shrine wall unseen.",
        in_game_time_delta_minutes=5,
        visible_changes=["Guard attention shifts toward the road."],
        hidden_changes=["The listener in the crypt hears the bell rope move."],
        new_clocks=[{"id": "tide", "name": "The turning tide", "segments": 6}],
        new_hooks=["Find the bell-keeper before the tide turns."],
        scene_title="The shrine wall",
    )
    assert result["ok"], result
    state = service.store.read_state()
    assert state.in_game_minutes == 5
    assert state.clocks[0].id == "tide"
    assert state.scene.hooks == ["Find the bell-keeper before the tide turns."]

    advanced = service.scene_commit("The tide rises.", clock_updates={"tide": 6})
    assert any("full" in warning for warning in advanced["warnings"])


def test_scene_commit_anchors_and_titles_a_fresh_scene_from_authored_canon(
    service: GameService, tmp_path,
):
    """Test scene commit anchors and titles a fresh scene from authored canon.
    """
    first = service.scene_commit("The dark holds its breath.")
    assert first["ok"], first
    scene = service.store.read_state().scene
    assert scene.location_id == "the-eel-market"
    assert scene.title == "The Eel Market"

    # Both fills are durable and never re-fire over established values.
    service.scene_commit("The tide creeps back in.")
    scene = service.store.read_state().scene
    assert scene.location_id == "the-eel-market"
    assert scene.title == "The Eel Market"

    # Explicit arguments still win over both fills and existing values.
    moved = service.scene_commit(
        "The party climbs to the shrine.",
        location_id="the-road-shrine",
        scene_title="The road shrine at dusk",
    )
    assert moved["ok"], moved
    scene = service.store.read_state().scene
    assert scene.location_id == "the-road-shrine"
    assert scene.title == "The road shrine at dusk"

    # A world with no authored locations gets neither fill: the scene stays
    # exactly as the model left it rather than gaining a meaningless anchor.
    import shutil

    bare_root = tmp_path / "bare"
    bare_root.mkdir()
    shutil.copytree(service.store.rules_dir, bare_root / "rules")
    bare = GameService(bare_root)
    bare.store.initialize(title="Bare")
    before = bare.store.read_state().scene.location_id
    result = bare.scene_commit("Nothing is anywhere yet.")
    assert result["ok"], result
    scene = bare.store.read_state().scene
    assert scene.location_id == before
    assert scene.title == ""


def test_scene_commit_drops_blank_entries_from_narrator_lists(service: GameService):
    """A narrator that pads a list with empty strings must not corrupt canon."""
    result = service.scene_commit(
        "The listener signals twice.",
        visible_changes=["A lamp flashes in the customs house.", "", "   "],
        hidden_changes=["", "Ammet has seen the party's faces.", ""],
        new_hooks=["", "Answer the Choir before the tide."],
        exits=["the-road-shrine", ""],
    )
    assert result["ok"], result
    scene = service.store.read_state().scene
    assert "" not in scene.visible_facts
    assert "" not in scene.hidden_facts
    assert "" not in scene.hooks
    assert scene.exits == ["the-road-shrine"]
    assert scene.hidden_facts == ["Ammet has seen the party's faces."]


def test_scene_commit_locks_and_unlocks_a_typed_barrier(service: GameService):
    """Test scene commit locks and unlocks a typed barrier.
    """
    locked = service.scene_commit(
        "The party finds the tower door barred.",
        object_updates={
            "tower-door": {"state": "locked", "note": "an iron slide bolt, not a lock"}
        },
    )
    assert locked["ok"], locked
    scene = service.store.read_state().scene
    assert scene.objects["tower-door"].state == "locked"
    assert scene.objects["tower-door"].note == "an iron slide bolt, not a lock"

    unlocked = service.scene_commit(
        "Someone throws the bolt from inside.",
        object_updates={"tower-door": {"state": "unlocked"}},
    )
    assert unlocked["ok"], unlocked
    scene = service.store.read_state().scene
    # The note is untouched: an update naming only a state must not erase detail an
    # earlier commit recorded.
    assert scene.objects["tower-door"].state == "unlocked"
    assert scene.objects["tower-door"].note == "an iron slide bolt, not a lock"


def test_scene_commit_rejects_an_unrecognized_object_state(service: GameService):
    result = service.scene_commit(
        "The party tries the door.",
        object_updates={"tower-door": {"state": "ajar"}},
    )
    assert result["ok"] is False
    assert result["error"] == "invalid_object_state"
    assert service.store.read_state().scene.objects == {}


def test_scene_commit_rejects_a_new_object_with_no_state(service: GameService):
    """A brand-new object has no prior state to fall back on, so state is mandatory."""
    result = service.scene_commit(
        "The party notices a chest.",
        object_updates={"old-chest": {"note": "iron-bound, no visible lock"}},
    )
    assert result["ok"] is False
    assert result["error"] == "empty_object_state"
    assert service.store.read_state().scene.objects == {}


def test_scene_commit_object_update_event_records_the_applied_state(service: GameService):
    result = service.scene_commit(
        "The party bars the door behind them.",
        object_updates={"tower-door": {"state": "barred", "note": "a heavy oak beam"}},
    )
    assert result["ok"], result
    event = service.store.read_events(limit=1)[0]
    assert event["object_updates"] == {
        "tower-door": {
            "id": "tower-door",
            "state": "barred",
            "note": "a heavy oak beam",
            "traversal_segments": None,
        }
    }


def test_traversal_clock_id_is_direction_normalized():
    """The same two locations always produce the same clock id, either order.
    """
    forward = traversal_clock_id("the-north-camp", "the-south-camp")
    backward = traversal_clock_id("the-south-camp", "the-north-camp")
    assert forward == backward == "travel-the-north-camp-the-south-camp"


def test_scene_commit_sizes_a_traversal_clock_to_the_typed_exits_segment_count(
    service: GameService,
):
    """The typed exit's own segment count is authoritative, not a caller's guess.

    A narrator turn (or a scenario harness) might pass any ``segments`` value when
    opening a traversal clock; when that clock's id names a typed exit that
    already carries ``traversal_segments``, the exit's own fixed count wins.
    """
    clock_id = traversal_clock_id("the-north-camp", "the-south-camp")
    typed = service.scene_commit(
        "A causeway links the two camps.",
        object_updates={clock_id: {"state": "open", "traversal_segments": 4}},
    )
    assert typed["ok"], typed

    opened = service.scene_commit(
        "The party starts across the causeway.",
        new_clocks=[{"id": clock_id, "name": "Crossing the causeway", "segments": 2}],
    )
    assert opened["ok"], opened
    clock = service.store.read_state().clocks[0]
    assert clock.id == clock_id
    assert clock.segments == 4, "the typed exit's own count, not the caller's guess"

    event = service.store.read_events(limit=1)[0]
    assert any("sized to the typed exit" in entry for entry in event["changes"])


def test_scene_commit_leaves_an_untyped_clocks_segments_alone(service: GameService):
    """A clock whose id names no typed exit keeps whatever the caller requested."""
    result = service.scene_commit(
        "The tide begins to turn.",
        new_clocks=[{"id": "tide", "name": "The turning tide", "segments": 6}],
    )
    assert result["ok"], result
    clock = service.store.read_state().clocks[0]
    assert clock.segments == 6


def test_reopening_an_unfinished_traversal_clock_from_either_direction_reuses_it(
    service: GameService,
):
    """Crossing the same path from either end opens exactly one clock.
    """
    forward_id = traversal_clock_id("the-north-camp", "the-south-camp")
    outbound = service.scene_commit(
        "The party starts north to south across the causeway.",
        new_clocks=[{"id": forward_id, "name": "Crossing the causeway", "segments": 4}],
        clock_updates={forward_id: 2},
    )
    assert outbound["ok"], outbound
    assert service.store.read_state().clocks[0].filled == 2

    backward_id = traversal_clock_id("the-south-camp", "the-north-camp")
    assert backward_id == forward_id

    return_trip = service.scene_commit(
        "Later, the party starts south to north across the same causeway.",
        new_clocks=[{"id": backward_id, "name": "Crossing the causeway", "segments": 4}],
    )
    assert return_trip["ok"], return_trip
    state = service.store.read_state()
    assert len(state.clocks) == 1, "a return trip must not open a second clock"
    assert state.clocks[0].id == forward_id
    assert state.clocks[0].filled == 2, "an unfinished crossing's own progress survives"


def test_reopening_a_completed_traversal_clock_resets_it_for_a_fresh_crossing(
    service: GameService,
):
    """Test reopening a completed traversal clock resets it for a fresh crossing.
    """
    clock_id = traversal_clock_id("the-north-camp", "the-south-camp")
    typed = service.scene_commit(
        "A causeway links the two camps.",
        object_updates={clock_id: {"state": "open", "traversal_segments": 3}},
    )
    assert typed["ok"], typed

    outbound = service.scene_commit(
        "The party finishes crossing north to south.",
        new_clocks=[{"id": clock_id, "name": "Crossing the causeway", "segments": 3}],
        clock_updates={clock_id: 3},
    )
    assert outbound["ok"], outbound
    finished = service.store.read_state().clocks[0]
    assert finished.filled == finished.segments == 3

    return_trip = service.scene_commit(
        "Later, the party starts the crossing south to north.",
        new_clocks=[{"id": clock_id, "name": "Crossing the causeway", "segments": 3}],
    )
    assert return_trip["ok"], return_trip
    state = service.store.read_state()
    assert len(state.clocks) == 1, "the same clock is reused, not a second one"
    reopened = state.clocks[0]
    assert reopened.id == clock_id
    assert reopened.filled == 0, "a finished crossing reopens at zero, not at its old count"
    assert reopened.segments == 3, "the segment count itself is unchanged by the reset"

    event = service.store.read_events(limit=1)[0]
    assert any("reopened for a fresh crossing" in entry for entry in event["changes"])


def test_clock_updates_against_a_completed_traversal_clock_restarts_it(service: GameService):
    """Test clock updates against a completed traversal clock restarts it.
    """
    clock_id = traversal_clock_id("the-north-camp", "the-south-camp")
    finished = service.scene_commit(
        "The party finishes crossing north to south.",
        new_clocks=[{"id": clock_id, "name": "Crossing the causeway", "segments": 3}],
        clock_updates={clock_id: 3},
    )
    assert finished["ok"], finished
    assert service.store.read_state().clocks[0].filled == 3

    # A negative delta, exactly the shape the live model sent: restarts at
    # zero, then applies the delta, rather than 3 + (-1) staying pinned near
    # the old maximum.
    negative = service.scene_commit(
        "Rill turns back toward the north camp.", clock_updates={clock_id: -1}
    )
    assert negative["ok"], negative
    assert service.store.read_state().clocks[0].filled == 0

    # From here the clock is back below its own segment count, so the
    # ordinary accumulate path applies again: unaffected by the restart logic.
    resumed = service.scene_commit(
        "The party presses on toward the north camp.", clock_updates={clock_id: 1}
    )
    assert resumed["ok"], resumed
    assert service.store.read_state().clocks[0].filled == 1

    # A positive delta against a (freshly re-)completed clock restarts the
    # same way, not merely a negative one.
    service.scene_commit("More progress.", clock_updates={clock_id: 2})
    assert service.store.read_state().clocks[0].filled == 3
    restarted_again = service.scene_commit(
        "A third crossing begins.", clock_updates={clock_id: 1}
    )
    assert restarted_again["ok"], restarted_again
    assert service.store.read_state().clocks[0].filled == 1


def test_rendered_scene_carries_no_trailing_whitespace(service: GameService):
    service.scene_commit(
        "The tide turns.",
        visible_changes=["The plank walk floods."],
        hidden_changes=["A second bell has cracked."],
    )
    text = service.store.scene_path.read_text(encoding="utf-8")
    offenders = [line for line in text.splitlines() if line != line.rstrip()]
    assert offenders == []


def test_npc_create_drops_blank_actions(service: GameService):
    result = service.npc_create(
        name="Reed Thug", level=1, motive=" rob the party ", actions=["swing a club", "", "   "]
    )
    assert result["ok"], result
    assert result["npc"]["actions"] == ["swing a club"]
    assert result["npc"]["motive"] == "rob the party"
    assert service.store.read_state().npcs["reed-thug"].actions == ["swing a club"]


def test_character_create_drops_blank_weapons(service: GameService, roller):
    result = make_character(service, roller, name="Bare", weapons=("axe", "", "  "))
    assert result["sheet"]["weapons"] == ["axe"]
    assert service.store.read_character("bare").weapons == ["axe"]


def test_a_blank_weapon_cannot_satisfy_the_parry_requirement(service: GameService, roller):
    """A blank string is truthy in a list and must not pass the held-object gate."""
    make_character(service, roller, name="Bare", weapons=("", "   "))
    npc = service.npc_create(name="Reed Thug", level=1, motive="rob")
    result = service.combat_defend("bare", npc["npc_id"], method="parry")
    assert result["ok"] is False
    assert result["error"] == "nothing_to_parry_with"


def test_session_close_drops_blank_open_hooks(service: GameService, roller):
    make_fighter(service, roller)
    result = service.session_close(
        "Night falls",
        "The party limps home.",
        open_hooks=["The Choir knows their faces.", "", "   "],
    )
    assert result["ok"], result
    assert service.store.read_state().scene.hooks == ["The Choir knows their faces."]
    summary = Path(result["summary_path"]).read_text(encoding="utf-8")
    assert "\n- \n" not in summary
    assert [line for line in summary.splitlines() if line != line.rstrip()] == []


def test_session_close_writes_no_file_when_the_transaction_fails(
    service: GameService, roller
):
    """An advisory summary must never claim a session that did not close."""
    make_fighter(service, roller)
    service.store.manifest_path.write_text("::: not valid yaml :::\n", encoding="utf-8")

    result = service.session_close("Broken", "This must not be recorded.")
    assert result["ok"] is False
    assert result["error"] == "invalid_manifest_file"
    assert not (service.store.summaries_dir / "session-001.md").exists()
    assert service.store.read_state().session == 1


def test_a_blank_summary_cannot_erase_committed_canon(service: GameService):
    """A whitespace string is truthy and must not overwrite the scene summary."""
    service.scene_commit("The party reaches the shrine wall unseen.")
    result = service.scene_commit("   ")
    assert result["ok"] is False
    assert result["error"] == "empty_public_summary"
    assert (
        service.store.read_state().scene.summary == "The party reaches the shrine wall unseen."
    )


def test_scene_commit_strips_the_scene_title(service: GameService):
    service.scene_commit("The tide turns.", scene_title="  Low water  ")
    assert service.store.read_state().scene.title == "Low water"
    text = service.store.scene_path.read_text(encoding="utf-8")
    assert "# Low water" in text


def test_session_close_refuses_a_blank_title_or_summary(service: GameService, roller):
    make_fighter(service, roller)
    for title, summary in ((" ", "Real summary."), ("Real title", "   ")):
        result = service.session_close(title, summary)
        assert result["ok"] is False
        assert result["error"] == "empty_session_record"
    assert service.store.read_state().session == 1
    assert not (service.store.summaries_dir / "session-001.md").exists()


def test_npc_create_validates_and_strips_the_location(service: GameService):
    bad = service.npc_create(name="Reed Thug", level=1, location_id="   ../../etc  ")
    assert bad["ok"] is False
    assert bad["error"] == "location_not_found"
    assert service.store.read_state().npcs == {}

    good = service.npc_create(name="Reed Thug", level=1, location_id="  the-eel-market  ")
    assert good["ok"], good
    assert good["npc"]["location_id"] == "the-eel-market"


def test_a_blank_clock_name_is_refused(service: GameService):
    result = service.scene_commit(
        "The tide turns.", new_clocks=[{"id": "tide", "name": "   ", "segments": 6}]
    )
    assert result["ok"] is False
    assert result["error"] == "empty_clock_name"
    assert service.store.read_state().clocks == []


def test_the_rendered_scene_never_loses_its_fallbacks(service: GameService):
    """Every rendered heading and bullet must carry real text."""
    service.scene_commit(
        "The bell tower stands silent.",
        new_clocks=[{"id": "tide", "name": "  The turning tide  ", "segments": 6}],
    )
    text = service.store.scene_path.read_text(encoding="utf-8")
    assert "# " in text and "\n# \n" not in text
    assert "- The turning tide (0/6)" in text
    assert [line for line in text.splitlines() if line != line.rstrip()] == []


def test_scene_commit_rejects_an_unknown_clock(service: GameService):
    result = service.scene_commit("Nothing happens.", clock_updates={"missing": 1})
    assert result["ok"] is False
    assert result["error"] == "clock_not_found"


def test_scene_commit_rejects_time_running_backwards(service: GameService):
    result = service.scene_commit("Rewind.", in_game_time_delta_minutes=-5)
    assert result["ok"] is False
    assert result["error"] == "invalid_time_delta"


def test_scene_commit_rejects_an_unknown_location(service: GameService):
    result = service.scene_commit("The party walks into nowhere.", location_id="atlantis")
    assert result["ok"] is False
    assert result["error"] == "location_not_found"


def test_scene_commit_refuses_a_location_change_while_combat_is_active(
    service: GameService, roller
):
    """An abandoned fight must never carry along silently into the next scene (M4)."""
    make_fighter(service, roller)
    open_fight(service, roller)
    result = service.scene_commit(
        "The party bolts down the tunnel.", location_id="the-road-shrine"
    )
    assert result["ok"] is False
    assert result["error"] == "combat_blocks_location_change"
    assert any("combat_close" in step for step in result["allowed_next_steps"])
    # The refused move changed nothing: the scene stays put and the fight stays open.
    state = service.store.read_state()
    assert state.scene.location_id == ""
    assert state.combat.active is True


def test_scene_commit_moves_the_party_after_combat_close(service: GameService, roller):
    """Once the fight is closed, the same location change the guard refused succeeds."""
    make_fighter(service, roller)
    open_fight(service, roller)
    refused = service.scene_commit(
        "The party bolts down the tunnel.", location_id="the-road-shrine"
    )
    assert refused["ok"] is False
    closed = service.combat_close("fled")
    assert closed["ok"], closed
    moved = service.scene_commit(
        "The party bolts down the tunnel.", location_id="the-road-shrine"
    )
    assert moved["ok"], moved
    assert service.store.read_state().scene.location_id == "the-road-shrine"


def test_scene_commit_location_change_is_unaffected_without_active_combat(
    service: GameService,
):
    """No fight running: the pre-existing, unguarded location-change path is untouched."""
    result = service.scene_commit(
        "The party crosses the mudflat.", location_id="the-road-shrine"
    )
    assert result["ok"], result
    assert service.store.read_state().scene.location_id == "the-road-shrine"


def test_scene_commit_with_the_same_location_is_unaffected_by_active_combat(
    service: GameService, roller
):
    """Re-affirming the current location mid-fight is not a location change; only an
    actual move is guarded."""
    placed = service.scene_commit("The party enters the shrine.", location_id="the-road-shrine")
    assert placed["ok"], placed
    make_fighter(service, roller)
    open_fight(service, roller)
    result = service.scene_commit("The fight rages on.", location_id="the-road-shrine")
    assert result["ok"], result
    assert service.store.read_state().scene.location_id == "the-road-shrine"


def test_session_close_writes_a_summary_awards_stories_and_backs_up(
    service: GameService, roller
):
    make_fighter(service, roller)
    result = service.session_close(
        session_title="The Ashen Bell",
        public_summary="The party found the bell-keeper's body beneath the shrine.",
        character_stories_awarded={"mara": 1},
        open_hooks=["The Choir Below knows the party's faces."],
        next_intention="Return to Vey before the tide.",
    )
    assert result["ok"], result
    assert Path(result["summary_path"]).is_file()
    assert Path(result["backup_path"]).is_file()
    assert service.store.read_state().session == 2
    assert service.store.read_manifest().session == 2
    assert service.store.read_character("mara").stories == 1
    assert result["advancement_eligible"][0]["eligible_level"] == 2
    assert (service.store.logs_dir / "session-002.md").is_file()


def test_session_close_clears_session_conditions(service: GameService, roller):
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    drop_to_helpless(service, roller, npc["npc_id"])
    roller.queue(4, 2)
    service.helpless_roll("mara")
    assert service.store.read_character("mara").conditions

    service.session_close("Night falls", "The party limps home.", accept_uncommitted=True)
    assert service.store.read_character("mara").conditions == []
