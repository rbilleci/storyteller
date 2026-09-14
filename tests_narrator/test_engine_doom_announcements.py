"""Pin the Doom-die announcement injection, the sibling ``inject_missing_roll_
announcements`` (``tests_narrator/test_engine_roll_announcements.py``) already gets.

The fixtures below through ``test_doom_facts_reads_the_nested_initiative_shape_from_
combat_start`` predate a correction worth naming: they put a top-level ``doom`` list
straight onto a ``combat_attack``/``combat_defend`` payload, on the strength of that
same ``events.jsonl`` record -- but that record is ``bsh_mcp.service``'s internal
``commit_payload``, written to the audit log through ``_doom_log``, not the *return
envelope* a tool call actually hands back to this narrator. Checked directly against
a live ``combat_attack``: the rules engine rolled Doom correctly on a
genuine repeat action, but the return envelope carried no ``doom`` key at all -- the
same real roll rode under ``repeat_action_doom`` instead, a single dict. Every real
tool call hits ``doom_facts``'s newer branches below (``repeat_action_doom``,
``critical_failure_doom``, ``called_on_doom``, ``individual``), covered from
``test_doom_facts_reads_a_repeat_action_roll_from_a_real_combat_attack_envelope``
onward; the ``doom``-list and bare-``doom``-dict branches stay covered too, since
``doom_facts`` still accepts either shape defensively, but no tool's own return
envelope produces the list shape today. These are pure functions with no engine,
agent, or campaign dependency, tested directly in the same mold the roll-announcement
sibling uses.
"""

from __future__ import annotations

from narrator.engine import (
    DOOM_ANNOUNCED_TOOLS,
    doom_facts,
    format_doom_announcement,
    inject_missing_doom_announcements,
)


def _doom_entry(previous_die="d6", current_die="d6", total=4, downgraded=False, depleted=False):
    return {
        "mode": "roll",
        "previous_die": previous_die,
        "current_die": current_die,
        "downgraded": downgraded,
        "depleted": depleted,
        "roll": {"notation": "1d6", "dice": [total], "selected": total, "modifier": 0, "total": total, "edge": "single"},
    }


def _held_payload(character="vessa", total=4):
    return {
        "ok": True, "attacker_id": character, "outcome": "failure",
        "doom": [_doom_entry(total=total)],
    }


def _downgraded_payload(character="vessa", total=1):
    return {
        "ok": True, "attacker_id": character, "outcome": "success",
        "doom": [_doom_entry(current_die="d4", total=total, downgraded=True)],
    }


def test_doom_facts_reads_a_held_roll_from_a_combat_attack_envelope():
    """Test doom facts reads a held roll from a combat attack envelope.
    """
    facts = doom_facts("combat_attack", _held_payload())
    assert facts == (
        {
            "tool": "combat_attack", "character": "vessa", "previous_die": "d6",
            "current_die": "d6", "total": 4, "downgraded": False, "depleted": False,
        },
    )


def test_doom_facts_reads_a_downgrade_from_a_combat_attack_envelope():
    """Test doom facts reads a downgrade from a combat attack envelope.
    """
    facts = doom_facts("combat_attack", _downgraded_payload())
    assert facts == (
        {
            "tool": "combat_attack", "character": "vessa", "previous_die": "d6",
            "current_die": "d4", "total": 1, "downgraded": True, "depleted": False,
        },
    )


def test_doom_facts_prefers_defender_id_over_attacker_id():
    """The roller on a ``combat_defend`` call is the defender, never the attacker --
    the same precedence ``roll_facts`` documents for the identical reason."""
    payload = {
        "ok": True, "defender_id": "vessa", "attacker_id": "rade",
        "doom": [_doom_entry()],
    }
    facts = doom_facts("combat_defend", payload)
    assert facts[0]["character"] == "vessa"


def test_doom_facts_reads_the_nested_initiative_shape_from_combat_start():
    """``combat_start`` carries no top-level ``doom``; a critical failure on
    initiative nests one ``_mandatory_doom`` dict per character instead."""
    payload = {
        "ok": True,
        "initiative": [
            {
                "character_id": "vessa", "name": "Vessa",
                "roll": {"total": 20}, "outcome": "critical_failure",
                "doom": _doom_entry(current_die="d4", total=1, downgraded=True),
            }
        ],
    }
    facts = doom_facts("combat_start", payload)
    assert facts == (
        {
            "tool": "combat_start", "character": "vessa", "previous_die": "d6",
            "current_die": "d4", "total": 1, "downgraded": True, "depleted": False,
        },
    )


def test_doom_facts_reads_a_repeat_action_roll_from_a_real_combat_attack_envelope():
    """``combat_attack``'s actual return envelope, confirmed by calling the real
    service twice in one open turn: no top-level ``doom`` key at all, the roll
    riding under ``repeat_action_doom`` instead, a single dict."""
    payload = {
        "ok": True, "attacker_id": "vessa", "outcome": "success",
        "repeat_action_doom": _doom_entry(total=5),
        "critical_failure_doom": None,
    }
    facts = doom_facts("combat_attack", payload)
    assert facts == (
        {
            "tool": "combat_attack", "character": "vessa", "previous_die": "d6",
            "current_die": "d6", "total": 5, "downgraded": False, "depleted": False,
        },
    )


def test_doom_facts_reads_both_repeat_and_critical_doom_from_one_combat_attack_call():
    """A repeated attack that also crit-fails rolls Doom twice in one call --
    two independent real rolls, both owed their own announcement."""
    payload = {
        "ok": True, "attacker_id": "vessa", "outcome": "critical_failure",
        "repeat_action_doom": _doom_entry(total=3),
        "critical_failure_doom": _doom_entry(current_die="d4", total=1, downgraded=True),
    }
    facts = doom_facts("combat_attack", payload)
    assert len(facts) == 2
    assert facts[0]["total"] == 3 and not facts[0]["downgraded"]
    assert facts[1]["total"] == 1 and facts[1]["downgraded"]


def test_doom_facts_reads_a_bare_dict_critical_failure_from_attribute_test():
    """``attribute_test``'s own critical-failure Doom rides under ``doom`` too, but
    as a bare dict, never a list -- the shape its actual return envelope uses."""
    payload = {"ok": True, "character_id": "mara", "doom": _doom_entry(total=1)}
    facts = doom_facts("attribute_test", payload)
    assert facts[0]["character"] == "mara"
    assert facts[0]["total"] == 1


def test_doom_facts_reads_a_voluntary_call_on_doom_from_attribute_test():
    """Calling on Doom deliberately is still a real roll a tool result carries,
    under its own ``called_on_doom`` key."""
    payload = {"ok": True, "character_id": "mara", "called_on_doom": _doom_entry(total=5)}
    facts = doom_facts("attribute_test", payload)
    assert facts[0]["character"] == "mara"
    assert facts[0]["total"] == 5


def test_doom_facts_reads_a_critical_failure_from_combat_defend():
    """``combat_defend`` carried no Doom-related key in its return envelope at all
    until this fix; a critical defence failure rolls Doom the same as an attack."""
    payload = {
        "ok": True, "defender_id": "vessa", "attacker_id": "rade",
        "critical_failure_doom": _doom_entry(current_die="d4", total=1, downgraded=True),
    }
    facts = doom_facts("combat_defend", payload)
    assert facts[0]["character"] == "vessa"
    assert facts[0]["downgraded"] is True


def test_doom_facts_reads_the_per_participant_individual_shape_from_group_test():
    """``group_test`` nests each participant's own Doom the same way ``combat_start``
    nests each character's initiative Doom -- a group test can roll Doom for more
    than one participant in a single call."""
    payload = {
        "ok": True,
        "individual": [
            {"character_id": "mara", "name": "Mara", "outcome": "success"},
            {
                "character_id": "vessa", "name": "Vessa", "outcome": "critical_failure",
                "doom": _doom_entry(current_die="d4", total=1, downgraded=True),
            },
        ],
    }
    facts = doom_facts("group_test", payload)
    assert facts == (
        {
            "tool": "group_test", "character": "vessa", "previous_die": "d6",
            "current_die": "d4", "total": 1, "downgraded": True, "depleted": False,
        },
    )


def test_doom_facts_ignores_a_tool_outside_the_announcement_policy():
    """``doom_roll`` already narrates its own outcome through ``narration_facts``;
    injecting a second, differently-worded line would duplicate a fact the model
    was already handed truthfully."""
    payload = {"ok": True, "character_id": "vessa", "doom": [_doom_entry()]}
    assert doom_facts("doom_roll", payload) == ()
    assert "doom_roll" not in DOOM_ANNOUNCED_TOOLS


def test_doom_facts_returns_nothing_for_a_failed_or_empty_result():
    assert doom_facts("combat_attack", {"ok": False, "doom": [_doom_entry()]}) == ()
    assert doom_facts("combat_attack", {"ok": True, "doom": []}) == ()
    assert doom_facts("combat_attack", {"ok": True}) == ()
    assert doom_facts("combat_attack", {}) == ()


def test_doom_facts_drops_a_roll_less_entry():
    """A restore or an already-depleted die contributes no ``roll`` key and no
    total to announce."""
    payload = {"ok": True, "character_id": "vessa", "doom": [{"mode": "restore", "previous_die": "d4", "current_die": "d6", "downgraded": False, "depleted": False}]}
    assert doom_facts("combat_attack", payload) == ()


def test_format_doom_announcement_states_a_held_roll():
    fact = {"previous_die": "d6", "current_die": "d6", "total": 4, "downgraded": False, "depleted": False}
    assert format_doom_announcement(fact, "Vessa") == "Vessa rolls the Doom die (d6): rolled 4, the die holds."


def test_format_doom_announcement_states_a_downgrade():
    fact = {"previous_die": "d6", "current_die": "d4", "total": 1, "downgraded": True, "depleted": False}
    assert format_doom_announcement(fact, "Vessa") == "Vessa rolls the Doom die (d6): rolled 1, the die drops to d4."


def test_format_doom_announcement_prioritises_depleted_over_downgraded():
    """A depleting roll can carry ``downgraded: True`` too -- stepping into the
    spent state is itself a step down -- and "the die is spent" is the fact a
    table needs, not "the die drops to spent"."""
    fact = {"previous_die": "d4", "current_die": "spent", "total": 1, "downgraded": True, "depleted": True}
    assert format_doom_announcement(fact, "Vessa") == "Vessa rolls the Doom die (d4): rolled 1, the Doom die is spent."


def test_injection_prepends_one_line_per_fact_after_any_roll_line():
    narration = "Vessa's weapon bites deep. The fishmonger collapses, dead."
    fact = {"tool": "combat_attack", "character": "vessa", "previous_die": "d6", "current_die": "d4", "total": 1, "downgraded": True, "depleted": False}
    amended, injected = inject_missing_doom_announcements(narration, (fact,), {"vessa": "Vessa"})
    assert amended == (
        "Vessa rolls the Doom die (d6): rolled 1, the die drops to d4.\n\n"
        "Vessa's weapon bites deep. The fishmonger collapses, dead."
    )
    assert injected == 1


def test_injection_is_a_no_op_on_no_facts():
    narration = "Nothing rolled this turn."
    amended, injected = inject_missing_doom_announcements(narration, (), {})
    assert amended == narration
    assert injected == 0


def test_injection_falls_back_to_a_readable_transform_of_an_unresolved_id():
    fact = {"tool": "combat_attack", "character": "reed-thug", "previous_die": "d6", "current_die": "d6", "total": 4, "downgraded": False, "depleted": False}
    amended, injected = inject_missing_doom_announcements("The fight rages on.", (fact,), {})
    assert amended.startswith("Reed Thug rolls the Doom die (d6): rolled 4, the die holds.\n\n")
    assert injected == 1
