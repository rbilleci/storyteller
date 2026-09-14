"""A person is someone present in the fiction with no mechanical record -- a clerk, a
trader, someone the party only talks to. ``NPC`` carries hit points because
``combat_start`` needs them, so before this field such a person lived only in
``visible_facts`` prose and ``present_npcs`` read ``None recorded`` beside five facts
about \"a Salt Magistrate clerk\". These pins cover the one tool path (``scene_commit``
``persons``), the one promotion path (``npc_create`` on the same slug), the authored
seeding from a location's ``visible_entities``, the render, the digest keying, and
the trust boundary the record deliberately does not cross.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from conftest import REPO_ROOT, ScriptedRoller, make_character

from bsh_mcp.models import ScenePerson
from bsh_mcp.service import GameService
from bsh_mcp.store import PERSONS_RENDER_CAP, CampaignStore

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from new_campaign import STARTING_SCENE, seed_authored_persons  # noqa: E402


def _persons(service: GameService) -> dict[str, ScenePerson]:
    return service.store.read_state().scene.persons


def _seq(result: dict) -> int:
    """The audit sequence a success envelope names (``evt-000007`` -> 7)."""
    return int(result["event_id"].rsplit("-", 1)[1])


def _last_event(service: GameService) -> dict:
    return service.store.read_events(limit=1)[-1]


def test_scene_commit_records_a_person_under_a_server_allocated_id(service: GameService):
    result = service.scene_commit(
        "A Salt Magistrate clerk walks toward Rill.",
        persons=[{"name": "Salt Magistrate clerk", "role": "counts barrels"}],
    )
    assert result["ok"] is True
    persons = _persons(service)
    assert list(persons) == ["salt-magistrate-clerk"]
    person = persons["salt-magistrate-clerk"]
    assert person.name == "Salt Magistrate clerk"
    assert person.role == "counts barrels"
    assert person.source == "model"  # stamped by the server, never passed by the model
    assert person.first_event == person.last_event == _seq(result)
    assert _last_event(service)["persons_added"] == ["salt-magistrate-clerk"]


def test_a_person_named_again_updates_monotonically(service: GameService):
    first = service.scene_commit("The clerk appears.", persons=[{"name": "Salt Magistrate clerk"}])
    second = service.scene_commit(
        "The clerk counts barrels.",
        persons=[{"name": "salt magistrate clerk", "role": "counts barrels"}],
    )
    person = _persons(service)["salt-magistrate-clerk"]
    assert person.first_event == _seq(first)
    assert person.last_event == _seq(second)
    assert person.role == "counts barrels"  # an empty role fills

    service.scene_commit("The clerk frowns.", persons=[{"name": "Salt Magistrate clerk", "role": "spy"}])
    assert _persons(service)["salt-magistrate-clerk"].role == "counts barrels"  # never overwritten
    assert len(_persons(service)) == 1


def test_a_bare_string_entry_is_taken_as_a_name_and_blanks_are_dropped(service: GameService):
    service.scene_commit("Two traders argue.", persons=["Vell the eel-seller", "", "   ", {"name": ""}])
    assert list(_persons(service)) == ["vell-the-eel-seller"]


def test_a_person_naming_a_recorded_npc_warns_and_writes_nothing(service: GameService):
    service.npc_create(name="Rade", level=1)
    result = service.scene_commit("Rade waves.", persons=[{"name": "rade"}, {"name": "Rade"}])
    assert result["ok"] is True
    assert _persons(service) == {}
    assert any("present_npcs" in warning for warning in result.get("warnings", []))


def test_npc_create_promotes_the_person_with_the_same_slug(service: GameService):
    """The clerk is introduced as a person, then a fight needs him as an NPC: one
    identifier, one record, the role carried into the NPC's notes, the person gone
    from ``scene.persons``. Before this slice ``npc_create`` knew nothing of persons
    and would have left both records standing under one id."""
    service.scene_commit(
        "A clerk approaches.", persons=[{"name": "Salt Magistrate clerk", "role": "counts barrels"}]
    )
    result = service.npc_create(name="Salt Magistrate clerk", level=2)
    assert result["npc_id"] == "salt-magistrate-clerk"
    state = service.store.read_state()
    assert "salt-magistrate-clerk" not in state.scene.persons
    npc = state.npcs["salt-magistrate-clerk"]
    assert npc.notes == "counts barrels"
    assert "salt-magistrate-clerk" in state.scene.present_npcs
    assert _last_event(service)["promoted_from_person"] is True


def test_npc_create_with_a_motive_keeps_the_motive_over_the_role(service: GameService):
    service.scene_commit("A clerk approaches.", persons=[{"name": "Clerk", "role": "counts barrels"}])
    service.npc_create(name="Clerk", level=1, motive="find the bell-keeper")
    npc = service.store.read_state().npcs["clerk"]
    assert npc.motive == "find the bell-keeper"
    assert npc.notes == ""


def test_npc_create_without_a_matching_person_promotes_nothing(service: GameService):
    service.scene_commit("A clerk approaches.", persons=[{"name": "Clerk"}])
    service.npc_create(name="Rade", level=1)
    state = service.store.read_state()
    assert list(state.scene.persons) == ["clerk"]
    assert _last_event(service)["promoted_from_person"] is False


def test_a_campaign_written_before_persons_validates_unchanged(service: GameService):
    """The ``objects`` precedent (tests/test_store.py): the raw pre-field shape, read
    through the real file layer, loads with the field defaulting empty."""
    service.scene_commit("The tide turns.", persons=[{"name": "Clerk"}])
    raw = json.loads(service.store.state_path.read_text(encoding="utf-8"))
    assert "persons" in raw["scene"]
    del raw["scene"]["persons"]
    service.store.state_path.write_text(json.dumps(raw), encoding="utf-8")
    state = CampaignStore(service.store.root).read_state()
    assert state.scene.persons == {}
    assert state.scene.summary == "The tide turns."


def test_the_render_lists_persons_newest_mention_first_and_caps_the_section(service: GameService):
    for index in range(PERSONS_RENDER_CAP + 2):
        service.scene_commit(f"Person {index} appears.", persons=[{"name": f"Person {index}"}])
    service.scene_commit("Person 0 returns.", persons=[{"name": "Person 0"}])
    rendered = service.store.scene_path.read_text(encoding="utf-8")
    section = rendered.split("## Other persons present", 1)[1].split("## ", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("- ")]
    assert len(lines) == PERSONS_RENDER_CAP
    assert lines[0] == "- person-0: Person 0"  # most recently mentioned first
    assert "- person-1:" not in section  # the oldest mention fell off the render
    assert len(_persons(service)) == PERSONS_RENDER_CAP + 2  # but stays in state
    # Bare id first, then the name, no role: the role stays in state only.
    assert "(" not in section


def test_an_empty_persons_section_renders_the_placeholder(service: GameService):
    service.scene_commit("Nothing new.")
    rendered = service.store.scene_path.read_text(encoding="utf-8")
    assert "## Other persons present\n\n- None recorded." in rendered


def test_seed_authored_persons_seeds_visible_entities_only(campaign_root: Path):
    """``world/locations/the-eel-market.md`` lists ``orso-pell`` visible and nothing
    hidden; a location that hides a listener must not seed him."""
    store = CampaignStore(campaign_root)
    store.initialize(title="Seeded")
    state = store.read_state()
    state.scene = STARTING_SCENE.model_copy(deep=True)
    assert seed_authored_persons(store, state) == ["orso-pell"]
    person = state.scene.persons["orso-pell"]
    assert person.source == "authored"
    assert person.name == "Orso Pell"
    assert person.role == "customs captain of Vey"
    assert seed_authored_persons(store, state) == []  # idempotent

    hidden = campaign_root / "world" / "locations" / "the-hideout.md"
    hidden.write_text(
        "---\nid: the-hideout\nname: The Hideout\nexits: []\nvisible_entities: []\n"
        "hidden_entities:\n  - ammet\n---\n\n## Public description\n\nA damp room.\n",
        encoding="utf-8",
    )
    state.scene.location_id = "the-hideout"
    assert seed_authored_persons(store, state) == []
    assert "ammet" not in state.scene.persons


def test_the_canon_digest_keys_the_npc_block_off_seeded_persons(campaign_root: Path):
    """An authored person shares its id with a ``world/npcs/index.yaml`` entry, so the
    digest pulls Orso Pell's motive and voice before any ``npc_create``."""
    from narrator import canon

    store = CampaignStore(campaign_root)
    store.initialize(title="Seeded")
    state = store.read_state()
    state.scene = STARTING_SCENE.model_copy(deep=True)
    seed_authored_persons(store, state)
    store.state_path.write_text(json.dumps(state.model_dump()), encoding="utf-8")
    store.scene_path.write_text(store.render_scene_markdown(state), encoding="utf-8")

    digest = canon.render_digest(campaign_root, campaign_root, 8000, 4000, 2000)
    assert "- id: orso-pell" in digest.text
    assert "motive: find out who is asking about the bell-keeper" in digest.text
    public = canon.render_digest(campaign_root, campaign_root, 8000, 4000, 2000, public=True)
    assert "- id: orso-pell" in public.text
    assert "motive:" not in public.text  # the planner's scope keeps id and name only


def test_the_trusted_scope_keeps_persons_out_of_the_presence_list(service: GameService):
    """The trust boundary the record keeps, as amended by Phase 3: a model-sourced
    person reaches ``TrustedScope.present_person_ids`` (routing) and never
    ``present_npc_ids``, which the hazard floor reads."""
    from narrator.interactions import read_trusted_scope

    service.scene_commit("A clerk approaches.", persons=[{"name": "Salt Magistrate clerk"}])
    scope = read_trusted_scope(service.store.root)
    assert scope.present_npc_ids == ()
    assert scope.present_person_ids == ("salt-magistrate-clerk",)
    service.npc_create(name="Salt Magistrate clerk", level=1)
    scope = read_trusted_scope(service.store.root)
    assert scope.present_npc_ids == ("salt-magistrate-clerk",)
    assert scope.present_person_ids == ()  # promoted: one identifier, one list


def test_scene_commit_refuses_a_non_list_persons_argument(service: GameService):
    """``@guard`` turns the CampaignError into the failure envelope; nothing is written."""
    result = service.scene_commit("x", persons={"name": "Clerk"})  # type: ignore[arg-type]
    assert result["ok"] is False
    assert result["error"] == "invalid_persons"
    assert _persons(service) == {}
    assert service.store.read_state().scene.summary != "x"


def test_an_epithet_label_converges_on_the_authored_id_and_promotes(service: GameService):
    """The Slice B soak recorded "Sera Vane, the bell-keeper" as
    ``sera-vane-the-bell-keeper`` beside the index's ``sera-vane``; a later
    ``npc_create("Sera Vane")`` would have allocated ``sera-vane`` and promoted
    nothing. Identity now converges on names: the label lands on the index id with
    the index name and role, and the promotion finds it."""
    service.scene_commit(
        "The bell-keeper steps out of the tower.",
        persons=[{"name": "Sera Vane, the bell-keeper"}],
    )
    persons = _persons(service)
    assert list(persons) == ["sera-vane"]
    assert persons["sera-vane"].name == "Sera Vane"
    assert persons["sera-vane"].role == "bell-keeper of the road shrine"  # from the index
    assert persons["sera-vane"].source == "model"  # how it arrived, not what it names

    result = service.npc_create(name="Sera Vane", level=1)
    assert result["npc_id"] == "sera-vane"
    assert _persons(service) == {}
    assert _last_event(service)["promoted_from_person"] is True


def test_a_label_carrying_a_recorded_npcs_name_writes_nothing(service: GameService):
    service.npc_create(name="Rade", level=1)
    result = service.scene_commit("Rade grumbles.", persons=["Rade, the fishmonger", "the fishmonger Rade"])
    assert _persons(service) == {}
    assert sum("present_npcs" in warning for warning in result["warnings"]) == 2


def test_a_label_carrying_a_party_members_name_writes_nothing(service: GameService, roller: ScriptedRoller):
    make_character(service, roller, name="Ossa")
    result = service.scene_commit("Ossa the barbarian spits.", persons=["Ossa the barbarian"])
    assert _persons(service) == {}
    assert any("party member" in warning for warning in result["warnings"])


def test_the_same_person_described_twice_is_one_record(service: GameService):
    first = service.scene_commit("A clerk approaches.", persons=["a Salt Magistrate clerk"])
    second = service.scene_commit(
        "The clerk frowns.", persons=["the Salt Magistrate clerk, frowning", "Salt Magistrate clerk"]
    )
    persons = _persons(service)
    assert list(persons) == ["salt-magistrate-clerk"]  # the leading article is not identity
    assert persons["salt-magistrate-clerk"].name == "Salt Magistrate clerk"
    assert persons["salt-magistrate-clerk"].first_event == _seq(first)
    assert persons["salt-magistrate-clerk"].last_event == _seq(second)


def test_a_short_name_does_not_match_inside_a_longer_one(service: GameService):
    """"Orso" alone is not "Orso Pell"; whole-word containment, not substring."""
    from bsh_mcp.service import resolve_person_label

    assert resolve_person_label("Orso", npcs={}, characters={}, persons={}, index={"orso-pell": "Orso Pell"}) == ("new", "orso")
    assert resolve_person_label("Orso Pell, captain", npcs={}, characters={}, persons={}, index={"orso-pell": "Orso Pell"}) == ("index", "orso-pell")
    assert resolve_person_label("ラデ、魚売り", npcs={"rade": "ラデ"}, characters={}, persons={}, index={}) == ("npc", "rade")
    assert resolve_person_label("the clerk", npcs={}, characters={}, persons={"clerk": "Clerk"}, index={}) == ("person", "clerk")
    assert resolve_person_label("An old ferryman", npcs={}, characters={}, persons={}, index={}) == ("new", "old-ferryman")


def test_session_close_reports_every_person_it_leaves_behind(service: GameService):
    service.scene_commit("A clerk approaches.", persons=["Salt Magistrate clerk"])
    service.scene_commit("A trader calls out.", persons=["Vell the eel-seller"])
    result = service.session_close(
        session_title="The silent bell", public_summary="The party asked about the keeper."
    )
    assert result["ok"] is True
    assert [person["id"] for person in result["persons"]] == ["vell-the-eel-seller", "salt-magistrate-clerk"]
    assert all(person["first_event"] <= person["last_event"] for person in result["persons"])
    summary = (service.store.summaries_dir / "session-001.md").read_text(encoding="utf-8")
    assert "## Persons present" in summary
    assert "- salt-magistrate-clerk: Salt Magistrate clerk (introduced at event" in summary
    assert _persons(service) != {}  # reported, never removed


def test_a_role_only_person_folds_into_the_authored_identity_once_named(service: GameService):
    """The iteration-2 soak left "the bell-keeper" recorded as ``bell-keeper``; when
    the party later hears "Sera Vane, the bell-keeper", the role-only record folds into
    ``sera-vane`` (whose authored role is "bell-keeper of the road shrine"), keeping
    the earlier ``first_event``. One person, one record."""
    first = service.scene_commit("A woman with a lamp: the bell-keeper.", persons=["the bell-keeper"])
    assert list(_persons(service)) == ["bell-keeper"]
    second = service.scene_commit("She gives her name.", persons=["Sera Vane, the bell-keeper"])
    persons = _persons(service)
    assert list(persons) == ["sera-vane"]
    assert persons["sera-vane"].first_event == _seq(first)
    assert persons["sera-vane"].last_event == _seq(second)
    assert persons["sera-vane"].role == "bell-keeper of the road shrine"
    # An authored person is never folded away, and an unrelated person stays.
    service.scene_commit("A clerk watches.", persons=["Salt Magistrate clerk"])
    service.scene_commit("Orso Pell strolls past.", persons=["Orso Pell, the customs captain"])
    assert sorted(_persons(service)) == ["orso-pell", "salt-magistrate-clerk", "sera-vane"]


def test_a_move_clears_model_persons_and_seeds_the_new_locations_authored_people(service: GameService):
    """A person the narration introduced belongs to the scene it was introduced in.
    Before this, the Eel Market clerk was still listed present at the Road Shrine."""
    # The fresh scene is ``unknown``, so the first located commit is itself a move and
    # seeds the market's authored person beside the clerk the model introduced.
    service.scene_commit("At the market.", location_id="the-eel-market", persons=["Salt Magistrate clerk"])
    assert sorted(_persons(service)) == ["orso-pell", "salt-magistrate-clerk"]
    assert _last_event(service)["persons_seeded"] == ["orso-pell"]
    service.scene_commit("The party walks to the shrine.", location_id="the-road-shrine")
    assert _persons(service) == {}
    event = _last_event(service)
    assert event["persons_cleared"] == ["orso-pell", "salt-magistrate-clerk"]
    # the-road-shrine's own visible_entities (none authored there) seed nothing; the
    # market's authored person seeds on the way back.
    service.scene_commit("Back to the market.", location_id="the-eel-market")
    persons = _persons(service)
    assert list(persons) == ["orso-pell"] and persons["orso-pell"].source == "authored"
    assert _last_event(service)["persons_seeded"] == ["orso-pell"]


def test_a_commit_without_a_move_keeps_every_person(service: GameService):
    service.scene_commit("At the market.", location_id="the-eel-market", persons=["Salt Magistrate clerk"])
    service.scene_commit("The clerk frowns.")
    service.scene_commit("Still here.", location_id="the-eel-market")
    assert sorted(_persons(service)) == ["orso-pell", "salt-magistrate-clerk"]
    assert _last_event(service)["persons_cleared"] == [] and _last_event(service)["persons_seeded"] == []


def test_a_person_recorded_on_both_sides_of_a_move_keeps_its_record(service: GameService):
    """An authored id present at both locations survives the move with its events."""
    store = service.store
    shrine = store.world_dir / "locations" / "the-road-shrine.md"
    text = shrine.read_text(encoding="utf-8").replace("visible_entities: []", "visible_entities:\n  - orso-pell")
    shrine.write_text(text, encoding="utf-8")
    service.scene_commit("At the market.", location_id="the-eel-market")
    first = service.scene_commit("Orso strolls.", persons=["Orso Pell"])
    service.scene_commit("To the shrine.", location_id="the-road-shrine")
    persons = _persons(service)
    assert list(persons) == ["orso-pell"]
    assert persons["orso-pell"].last_event == _seq(first)  # the record itself, not a fresh seed
    assert persons["orso-pell"].source == "authored"
    assert _last_event(service)["persons_cleared"] == [] and _last_event(service)["persons_seeded"] == []


def test_a_promoted_person_is_unaffected_by_a_move(service: GameService):
    service.scene_commit("At the market.", location_id="the-eel-market", persons=["Salt Magistrate clerk"])
    service.npc_create(name="Salt Magistrate clerk", level=1)
    service.scene_commit("To the shrine.", location_id="the-road-shrine")
    state = service.store.read_state()
    assert "salt-magistrate-clerk" in state.npcs and state.scene.persons == {}


# -- NPCs leave and join with the scene; a person may leave mid-scene -------------------


def test_npcs_leave_with_the_scene_they_belong_to_and_join_at_their_own(service: GameService):
    service.scene_commit("At the market.", location_id="the-eel-market")
    service.npc_create(name="Rade", level=1)  # present, no location given: belongs to the market
    service.npc_create(name="Keeper", level=1, location_id="the-road-shrine", present_in_scene=False)
    service.npc_create(name="Companion", level=1, present_in_scene=False)  # no location, then placed
    service.scene_commit("The companion joins.", present_npcs=["rade", "companion"])
    state = service.store.read_state()
    assert state.npcs["rade"].location_id == "the-eel-market"
    assert state.npcs["companion"].location_id == ""
    service.scene_commit("To the shrine.", location_id="the-road-shrine")
    state = service.store.read_state()
    assert state.scene.present_npcs == ["companion", "keeper"]  # rade left, the keeper joined, the companion stayed
    event = _last_event(service)
    assert event["npcs_left"] == ["rade"] and event["npcs_joined"] == ["keeper"]
    service.scene_commit("Back to the market.", location_id="the-eel-market")
    state = service.store.read_state()
    assert state.scene.present_npcs == ["companion", "rade"]


def test_a_corpse_stays_where_it_fell_and_a_fled_npc_does_not_rejoin(service: GameService):
    service.scene_commit("At the market.", location_id="the-eel-market")
    service.npc_create(name="Rade", level=1)
    service.npc_create(name="Runner", level=1)
    state = service.store.read_state()
    state.npcs["rade"].status = "dead"
    state.npcs["runner"].status = "fled"
    service.store.state_path.write_text(json.dumps(state.model_dump()), encoding="utf-8")
    service.scene_commit("To the shrine.", location_id="the-road-shrine")
    assert service.store.read_state().scene.present_npcs == []
    service.scene_commit("Back.", location_id="the-eel-market")
    assert service.store.read_state().scene.present_npcs == ["rade"]


def test_a_commit_without_a_move_moves_no_npc(service: GameService):
    service.scene_commit("At the market.", location_id="the-eel-market")
    service.npc_create(name="Rade", level=1)
    service.scene_commit("Rade grumbles.", location_id="the-eel-market")
    assert service.store.read_state().scene.present_npcs == ["rade"]
    assert _last_event(service)["npcs_left"] == [] and _last_event(service)["npcs_joined"] == []


def test_npc_create_keeps_an_explicit_or_absent_location(service: GameService):
    service.scene_commit("At the market.", location_id="the-eel-market")
    service.npc_create(name="Pilgrim", level=2, location_id="the-black-bell-crypt")
    service.npc_create(name="Ghost", level=1, present_in_scene=False)
    state = service.store.read_state()
    assert state.npcs["pilgrim"].location_id == "the-black-bell-crypt"
    assert state.npcs["ghost"].location_id == ""


def test_a_person_may_leave_mid_scene_by_name_and_an_unknown_name_only_warns(service: GameService):
    service.scene_commit("At the market.", location_id="the-eel-market", persons=["Vell the eel-seller", "Salt Magistrate clerk"])
    result = service.scene_commit(
        "Vell shutters her stall and walks off.", departed_persons=["Vell", "the eel-seller Vell", "nobody here"]
    )
    assert result["ok"] is True
    assert sorted(_persons(service)) == ["orso-pell", "salt-magistrate-clerk"]
    assert _last_event(service)["persons_departed"] == ["vell-the-eel-seller"]
    assert any("names no person present" in warning for warning in result["warnings"])
