"""Event-log completeness: every Doom die roll reaches ``events.jsonl``.
"""

from __future__ import annotations

from tools_shared import make_fighter, open_fight


def _last_event(service, tool: str) -> dict:
    """The most recent committed event for ``tool`` from ``events.jsonl``."""
    events = [e for e in service.store.read_events(limit=10) if e.get("tool") == tool]
    assert events, f"no committed {tool} event found"
    return events[-1]


def _faces(doom_log: list[dict]) -> list[int]:
    """The selected die face of each Doom roll a committed ``doom`` list holds."""
    return [entry["roll"]["selected"] for entry in doom_log if "roll" in entry]


def test_attribute_test_success_records_an_empty_doom_list(service, roller):
    """A test that rolls no Doom still carries the ``doom`` key as an empty list,
    so a reader distinguishes "no Doom rolled" from "Doom dropped from the record".
    """
    make_fighter(service, roller)
    roller.queue(5)  # STR 14, an unmodified 5 succeeds and triggers no Doom
    result = service.attribute_test("mara", "STR", "shoulder the door", "it opens", "it holds")
    assert result["outcome"] == "success"
    event = _last_event(service, "attribute_test")
    assert event["doom"] == []


def test_attribute_test_critical_failure_records_the_doom_face(service, roller):
    """A critical failure rolls mandatory Doom; the committed event must carry
    that rolled face, not only the return envelope.
    """
    make_fighter(service, roller)
    roller.queue(20, 1)  # d20 critical failure, then the Doom d6 shows 1
    result = service.attribute_test("mara", "WIS", "listen at the door", "it opens", "it holds")
    assert result["outcome"] == "critical_failure"
    envelope_face = result["doom"]["roll"]["selected"]
    event = _last_event(service, "attribute_test")
    assert envelope_face == 1
    assert envelope_face in _faces(event["doom"])


def test_attribute_test_called_on_doom_records_the_doom_face(service, roller):
    """Calling on Doom rolls the die before the test; the committed event must
    carry that called face.
    """
    make_fighter(service, roller)
    roller.queue(5, 15)  # Doom d6 shows 5 (the call), then the STR d20 shows 15
    result = service.attribute_test(
        "mara", "STR", "hold the gate", "it opens", "it holds", call_on_doom=True
    )
    envelope_face = result["called_on_doom"]["roll"]["selected"]
    event = _last_event(service, "attribute_test")
    assert envelope_face == 5
    assert envelope_face in _faces(event["doom"])


def test_combat_attack_repeat_action_records_the_doom_face(service, roller):
    """A second attack in one turn rolls mandatory Doom; the committed attack
    event must carry that repeat face, and the first attack must record no Doom.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    service.combat_begin_turn("mara")
    roller.queue(5, 2)  # first attack: d20 hit, damage die
    service.combat_attack("mara", npc["npc_id"])
    first_event = _last_event(service, "combat_attack")
    assert first_event["doom"] == []
    roller.queue(2, 5, 2)  # repeat Doom d6 shows 2, then d20 hit, then damage die
    second = service.combat_attack("mara", npc["npc_id"])
    envelope_face = second["repeat_action_doom"]["roll"]["selected"]
    second_event = _last_event(service, "combat_attack")
    assert envelope_face == 2
    assert envelope_face in _faces(second_event["doom"])


def test_combat_defend_critical_failure_records_the_doom_face(service, roller):
    """A critical defence failure rolls mandatory Doom; the committed defend event
    must carry that face. ``combat_defend`` used to omit Doom from its return
    envelope entirely -- the event was the only surface that could record it --
    until a narrator regression needed the return envelope itself to
    carry it too, the same as ``combat_attack``'s ``critical_failure_doom``.
    """
    make_fighter(service, roller)
    npc = open_fight(service, roller)
    roller.queue(20, 1)  # dodge d20 critical failure, then the Doom d6 shows 1
    result = service.combat_defend("mara", npc["npc_id"], method="dodge")
    assert result["outcome"] == "critical_failure"
    envelope_face = result["critical_failure_doom"]["roll"]["selected"]
    event = _last_event(service, "combat_defend")
    assert envelope_face == 1
    assert envelope_face in _faces(event["doom"])
    # The face 1 steps the Doom die down, which cross-checks the recorded value.
    assert service.store.read_character("mara").doom_die == "d4"


def test_group_test_still_records_a_participant_doom_face(service, roller):
    """group_test embeds each participant's mandatory Doom in its individual
    entries. This guards that the per-participant face stays in the record.
    """
    make_fighter(service, roller)
    roller.queue(20, 1)  # the lone participant critically fails, Doom d6 shows 1
    service.group_test(["mara"], "WIS", "hold the line", "it holds", "it breaks")
    event = _last_event(service, "group_test")
    participant_faces = []
    for entry in event["individual"]:
        doom = entry.get("doom")
        if doom and "roll" in doom:
            participant_faces.append(doom["roll"]["selected"])
    assert 1 in participant_faces
