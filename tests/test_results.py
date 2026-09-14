"""Coverage for the single result envelope every mechanics tool returns."""

from __future__ import annotations

from bsh_mcp import results


def test_event_id_none_for_no_sequence():
    assert results.event_id(None) is None


def test_event_id_formats_zero_padded():
    assert results.event_id(3) == "evt-000003"


def test_event_id_zero_is_a_valid_non_empty_id():
    assert results.event_id(0) == "evt-000000"


def test_event_id_widens_rather_than_truncates():
    assert results.event_id(1234567) == "evt-1234567"


def test_success_minimal_envelope_omits_optional_keys():
    envelope = results.success("did a thing")
    assert envelope == {
        "ok": True,
        "summary": "did a thing",
        "state_changes": [],
        "narration_facts": [],
        "warnings": [],
    }
    assert "event_id" not in envelope
    assert "roll" not in envelope
    assert "outcome" not in envelope


def test_success_includes_event_id_when_sequence_given():
    envelope = results.success("did a thing", sequence=7)
    assert envelope["event_id"] == "evt-000007"


def test_success_sequence_zero_still_sets_event_id():
    envelope = results.success("did a thing", sequence=0)
    assert envelope["event_id"] == "evt-000000"


def test_success_includes_roll_and_outcome_when_given():
    envelope = results.success("did a thing", roll={"die": 6, "result": 4}, outcome="success")
    assert envelope["roll"] == {"die": 6, "result": 4}
    assert envelope["outcome"] == "success"


def test_success_passes_through_state_changes_narration_facts_and_warnings():
    envelope = results.success(
        "did a thing",
        state_changes=["hp:-2"],
        narration_facts=["the blade bites deep"],
        warnings=["low on doses"],
    )
    assert envelope["state_changes"] == ["hp:-2"]
    assert envelope["narration_facts"] == ["the blade bites deep"]
    assert envelope["warnings"] == ["low on doses"]


def test_success_extra_kwargs_become_top_level_fields():
    envelope = results.success("did a thing", coin=3)
    assert envelope["coin"] == 3


def test_success_extra_kwarg_collision_overwrites_a_standard_key():
    # payload.update(extra) runs last in success(), so a caller-supplied kwarg that
    # collides with a standard key silently wins. Pinning existing behavior, not
    # prescribing it.
    envelope = results.success("did a thing", ok=False)
    assert envelope["ok"] is False


def test_failure_minimal_envelope():
    envelope = results.failure("bad_input", "nope")
    assert envelope == {
        "ok": False,
        "error": "bad_input",
        "message": "nope",
        "allowed_next_steps": [],
    }


def test_failure_with_allowed_next_steps():
    envelope = results.failure("bad_input", "nope", allowed_next_steps=["retry with a valid id"])
    assert envelope["allowed_next_steps"] == ["retry with a valid id"]
