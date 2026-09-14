"""Offline pins for the M2 and M17 soak instruments in ``narrator/soak_instruments.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

from narrator.delivery import TurnPost
from narrator.soak_instruments import (
    engine_turn_posts,
    find_raw_leaks,
    rest_declaration_turns,
    score_phantom_hp_claims,
    score_phantom_roll_claims,
    score_rest_handling,
    score_roll_announcements,
)


def _fake_config(**overrides) -> SimpleNamespace:
    """A config double carrying every delivery notice, each a distinct sentinel text.

    Distinct sentinels mean a test that mixes notices up fails loudly rather than
    passing by coincidence on two equal strings.
    """
    defaults = {
        "fault_notice": "NOTICE:fault",
        "withheld_notice": "NOTICE:withheld",
        "decision_fault_notice": "NOTICE:decision-fault",
        "decision_segment_notice": "NOTICE:decision-segment",
        "decision_declined_notice": "NOTICE:decision-declined",
        "risk_confirmation_notice": "NOTICE:risk-confirmation",
        "romance_boundary_notice": "NOTICE:romance-boundary",
        "trade_confirmation_notice": "NOTICE:trade-confirmation",
        "trade_completed_notice": "NOTICE:trade-completed",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# --- find_raw_leaks -----------------------------------------------------------------


def test_find_raw_leaks_is_silent_on_ordinary_narration():
    prose = (
        "The tide is out, but the mud is treacherous. Orso Pell counts barrels "
        "near the eel stalls while the bell tower stays silent."
    )
    assert find_raw_leaks(prose) == []


def test_find_raw_leaks_catches_the_provider_warning():
    """Test find raw leaks catches the provider warning.
    """
    leaks = find_raw_leaks(
        "reasoningContent is not supported in multi-turn conversations with the "
        "Chat Completions API."
    )
    assert any(leak.startswith("error:") for leak in leaks)


def test_find_raw_leaks_catches_a_traceback_and_an_exception_name():
    assert any(
        leak.startswith("error:")
        for leak in find_raw_leaks("Traceback (most recent call last):\n  File x")
    )
    assert any(
        leak.startswith("error:")
        for leak in find_raw_leaks("boom: ConnectionError while streaming")
    )


def test_find_raw_leaks_catches_unstripped_markup():
    leaks = find_raw_leaks('some text <|tool_call>{"name": "x"}</tool_call> more text')
    assert any(leak.startswith("markup:") for leak in leaks)


# --- engine_turn_posts ----------------------------------------------------------------


def _turn_narration(text: str) -> TurnPost:
    return TurnPost(kind="narration", origin="turn", model_text=text)


def _turn_notice(text: str, key: str = "") -> TurnPost:
    return TurnPost(kind="notice", origin="turn", notice_key=key, notice_text=text)


def _service_notice(text: str, key: str = "") -> TurnPost:
    return TurnPost(kind="notice", origin="service", notice_key=key, notice_text=text)


def test_engine_turn_posts_filters_by_recorded_origin():
    """Pre-engine notices drop; engine-backed narration AND notices survive.

    A real ``engine.run_turn`` outcome may itself post a fault or withheld notice --
    those carry ``origin="turn"`` and must survive the filter, not be mistaken for
    the pre-engine service branches.
    """
    config = _fake_config()
    post_log = [
        _turn_narration("Rill rolls DEX: 9 vs 14, success.\n\nYou pivot and bolt for the door."),
        _service_notice(config.risk_confirmation_notice, "risk_confirmation"),
        _service_notice(config.trade_confirmation_notice, "trade_confirmation"),
        _service_notice(config.romance_boundary_notice, "romance_boundary"),
        _turn_notice(config.fault_notice, "fault"),
        _turn_notice(config.withheld_notice, "withheld"),
        _turn_narration("Ossa rolls WIS: 2 vs 9, success."),
    ]

    assert engine_turn_posts(post_log) == [
        "Rill rolls DEX: 9 vs 14, success.\n\nYou pivot and bolt for the door.",
        config.fault_notice,
        config.withheld_notice,
        "Ossa rolls WIS: 2 vs 9, success.",
    ]


def test_engine_turn_posts_distinguishes_byte_identical_decision_fault_posts():
    """AUD-2-DECISION-FAULT-RESIDUAL-MISATTRIBUTION's undecidable case, retired.

    A guard-error/decision-recovery turn (post-engine, backs ``tool_log[0]``), a
    genuinely announced roll, a pre-engine social-choice divert posting the
    *byte-identical* ``decision_fault_notice`` with no backing call, then another
    announced roll. No text rule could tell the two identical posts apart --
    ``AmbiguousDecisionFaultAlignment`` refused to score the mix. ``TurnPost.origin``
    is recorded where the fact is decided, so the case is now simply decidable.
    """
    config = _fake_config()
    tool_log = [["attribute_test"], ["group_test"], ["combat_attack"]]
    post_log = [
        _turn_notice(config.decision_fault_notice, "decision_fault"),  # backs tool_log[0]
        _turn_narration("Ossa rolls WIS: 2 vs 9, success."),  # backs tool_log[1]
        _service_notice(config.decision_fault_notice, "decision_fault"),  # no backing call
        _turn_narration("Rill rolls STR: 9 vs 10, failure."),  # backs tool_log[2]
    ]

    aligned = engine_turn_posts(post_log)
    assert aligned == [
        config.decision_fault_notice,
        "Ossa rolls WIS: 2 vs 9, success.",
        "Rill rolls STR: 9 vs 10, failure.",
    ]
    assert len(aligned) == len(tool_log)

    # The decision-fault turn itself carried a roll-under tool call
    # (attribute_test) that never got announced -- decision_fault_notice replaced
    # the narration, exactly as fault_notice or withheld_notice would have. That is
    # a correct "unannounced" reading, not a misattribution.
    result = score_roll_announcements(tool_log, aligned)
    assert result["roll_turns"] == [1, 2, 3]
    assert result["unannounced_turns"] == [1]


def test_engine_turn_posts_keeps_a_withheld_turns_mechanical_lines_in_its_text():
    """A withheld turn's engine-authored roll lines ride above its notice text."""
    config = _fake_config()
    post = TurnPost(
        kind="notice",
        origin="turn",
        engine_lines=("Rill rolls DEX: 9 vs 14, success.",),
        notice_key="withheld",
        notice_text=config.withheld_notice,
    )

    assert engine_turn_posts([post]) == [
        "Rill rolls DEX: 9 vs 14, success.\n\n" + config.withheld_notice
    ]


# --- score_roll_announcements -----------------------------------------------------


def test_score_roll_announcements_matches_the_policy_shape_case_and_punctuation_tolerant():
    tool_log = [["campaign_status"], ["attribute_test"], ["group_test"], []]
    posts = [
        "You look around.",
        "Rill rolls DEX: 9 vs 14, success.\n\nYou pivot and bolt.",
        "ossa rolls wis: 2 vs. 9, critical success.",
        "Nothing happens.",
    ]
    result = score_roll_announcements(tool_log, posts)

    assert result == {
        "roll_turns": [2, 3],
        "roll_turn_count": 2,
        "unannounced_turns": [],
        "all_announced": True,
    }


def test_score_roll_announcements_flags_a_roll_under_turn_left_bare():
    tool_log = [["attribute_test"]]
    posts = ["You succeed at picking the lock, no number given."]

    result = score_roll_announcements(tool_log, posts)

    assert result["roll_turn_count"] == 1
    assert result["unannounced_turns"] == [1]
    assert result["all_announced"] is False


def test_score_roll_announcements_ignores_tools_that_carry_no_roll_versus_target():
    """usage_roll, doom_roll, and helpless_roll report a grade or a step, not a roll-under."""
    tool_log = [["usage_roll"], ["doom_roll"], ["helpless_roll"]]
    posts = ["torch dims", "doom advances", "you fall"]

    result = score_roll_announcements(tool_log, posts)

    assert result == {
        "roll_turns": [],
        "roll_turn_count": 0,
        "unannounced_turns": [],
        "all_announced": True,
    }


def test_score_roll_announcements_without_the_alignment_fix_would_misattribute_the_post():
    """Sabotage check: reproduce the exact drift a pre-engine notice causes.

    A failed persuasion roll (turn 1) provokes a hazard confirmation the next mention
    (turn 2), which never reaches ``engine.run_turn`` and so contributes zero
    ``tool_log`` entries; a group test then runs on the mention after that (turn 3).
    Indexing raw ``adapter.posted`` directly checks the roll-under turn's tool call
    against the *confirmation notice*, not against the reply that actually announces
    the roll -- exactly the bug ``engine_turn_posts`` exists to fix.
    """
    config = _fake_config()
    tool_log = [["attribute_test"], ["group_test"]]
    raw_posted = [
        "Rill rolls CHA: 15 vs 10, failure.\n\nThe clerk narrows his eyes.",
        config.risk_confirmation_notice,
        "Ossa rolls WIS: 2 vs 9, success.\nRill rolls WIS: 8 vs 14, success.",
    ]

    misaligned = score_roll_announcements(tool_log, raw_posted)
    assert misaligned["unannounced_turns"] == [2]
    assert misaligned["all_announced"] is False

    post_log = [
        _turn_narration(raw_posted[0]),
        _service_notice(raw_posted[1], "risk_confirmation"),
        _turn_narration(raw_posted[2]),
    ]
    aligned = score_roll_announcements(tool_log, engine_turn_posts(post_log))
    assert aligned["unannounced_turns"] == []
    assert aligned["all_announced"] is True


def test_score_phantom_roll_claims_detects_unbacked_successes():
    """Score phantom roll claims detects unbacked successes."""
    tool_log: list[list[str]] = [[] for _ in range(37)]
    posts = ["An empty cart rattles past." for _ in range(37)]
    posts[5] = (
        'A watch begins beside the warehouse.\n\nOssa rolls WIS: 14 vs 10, success.\n\nThe watch ends without incident.'
    )
    posts[25] = (
        'Ossa searches a storage cabinet.\n\nOssa rolls INT: 12 vs 10, success.\n\nThe inspection ends.'
    )

    result = score_phantom_roll_claims(tool_log, posts)

    assert result == {
        "phantom_roll_turns": [6, 26],
        "phantom_roll_turn_count": 2,
        "any_phantom_roll_claims": True,
        "turns_scored": 37,
    }


def test_score_phantom_roll_claims_ignores_a_turn_a_real_tool_call_backs():
    """A roll-verdict post backed by the matching tool call is not a phantom claim.

    Mirrors ``score_roll_announcements``'s own announced case: the same text, this
    time with ``attribute_test`` actually in the turn's tool log, must score clean --
    this function flags an *unbacked* claim, not every roll-shaped sentence.
    """
    tool_log = [["attribute_test"]]
    posts = ["Rill rolls DEX: 9 vs 14, success.\n\nYou pivot and bolt for the door."]

    result = score_phantom_roll_claims(tool_log, posts)

    assert result == {
        "phantom_roll_turns": [],
        "phantom_roll_turn_count": 0,
        "any_phantom_roll_claims": False,
        "turns_scored": 1,
    }


def test_score_phantom_roll_claims_ignores_ordinary_narration_with_no_verdict_claim():
    """A tool-free turn that never claims a roll verdict is not a phantom claim.

    The overwhelming majority of tool-free turns in a real session (ordinary prose,
    refusals, scene description) must not be flagged -- only a post that actually
    matches ``ROLL_ANNOUNCEMENT_PATTERN``'s announcement shape counts.
    """
    tool_log = [[], [], ["campaign_status"]]
    posts = [
        "You look around the market.",
        "I cannot resolve this action because the oar-case is not currently recorded.",
        "You check your inventory.",
    ]

    result = score_phantom_roll_claims(tool_log, posts)

    assert result["phantom_roll_turns"] == []
    assert result["any_phantom_roll_claims"] is False


def test_score_phantom_roll_claims_is_case_and_punctuation_tolerant_like_its_sibling():
    """The same shape tolerance ``score_roll_announcements`` documents applies here."""
    tool_log = [[]]
    posts = ["ossa rolls wis: 2 vs. 9, critical success."]

    result = score_phantom_roll_claims(tool_log, posts)

    assert result["phantom_roll_turns"] == [1]


def test_score_phantom_roll_claims_does_not_flag_a_combat_start_initiative_announcement():
    """`combat_start` rolls and audits a real WIS test per character, outside
    ``ROLL_UNDER_TOOLS``.
    """
    tool_log = [["combat_start"]]
    posts = ["Ossa rolls WIS: 11 vs 12, failure.\n\nThe cultists strike first."]

    result = score_phantom_roll_claims(tool_log, posts)

    assert result["phantom_roll_turns"] == []
    assert result["any_phantom_roll_claims"] is False


def test_score_phantom_roll_claims_flags_a_planning_narration_with_no_backing_call():
    """Test score phantom roll claims flags a planning narration with no backing call.
    """
    tool_log = [[]]
    posts = [
        'I will call an INT test for Ossa to inspect the cellar.\n\nOssa rolls INT: 17 vs 9, failure.'
    ]

    result = score_phantom_roll_claims(tool_log, posts)

    assert result["phantom_roll_turns"] == [1]


def test_score_phantom_roll_claims_uses_the_engine_turn_posts_alignment_contract():
    """The same misattribution hazard ``score_roll_announcements`` guards against.

    A pre-engine notice diverts one mention with no ``tool_log`` entry; indexing raw
    ``adapter.posted`` directly checks the *next* real turn's tool call against the
    notice text instead of its own reply, which could silently hide a phantom claim
    (or manufacture a false one) depending on what sits on either side. Aligning
    through ``engine_turn_posts`` first, exactly as ``score_roll_announcements``
    requires, restores the correct 1:1 correspondence.
    """
    config = _fake_config()
    tool_log = [["attribute_test"], []]
    raw_posted = [
        "Rill rolls CHA: 15 vs 10, failure.\n\nThe clerk narrows his eyes.",
        config.risk_confirmation_notice,
        "Ossa rolls WIS: 2 vs 9, success with no test called.",
    ]

    misaligned = score_phantom_roll_claims(tool_log, raw_posted)
    # Raw indexing checks tool_log[1] (empty -- no roll tool) against
    # raw_posted[1] (the notice, which does not match the pattern), so the real
    # unbacked claim at raw_posted[2] is never reached at all.
    assert misaligned["phantom_roll_turns"] == []

    post_log = [
        _turn_narration(raw_posted[0]),
        _service_notice(raw_posted[1], "risk_confirmation"),
        _turn_narration(raw_posted[2]),
    ]
    aligned = score_phantom_roll_claims(tool_log, engine_turn_posts(post_log))
    assert aligned["phantom_roll_turns"] == [2]


def test_score_phantom_hp_claims_detects_unbacked_recovery():
    """Score phantom hp claims detects unbacked recovery."""
    tool_log: list[list[str]] = [[] for _ in range(37)]
    posts = ["An empty cart rattles past." for _ in range(37)]
    posts[4] = (
        'The group pauses beside a dry wall.\n\nOssa recovers 5 hit points.\n\nThe pause is over.'
    )
    posts[10] = posts[4]

    result = score_phantom_hp_claims(tool_log, posts)

    assert result == {
        "phantom_hp_turns": [5, 11],
        "phantom_hp_turn_count": 2,
        "any_phantom_hp_claims": True,
        "turns_scored": 37,
    }


def test_score_phantom_hp_claims_ignores_a_turn_a_real_rest_call_backs():
    """A hit-point-figure post backed by the matching `rest` call is not a phantom claim."""
    tool_log = [["rest"]]
    posts = ["Ossa recovers 5 hit points (12/15). Rill recovers 3 hit points (9/9)."]

    result = score_phantom_hp_claims(tool_log, posts)

    assert result == {
        "phantom_hp_turns": [],
        "phantom_hp_turn_count": 0,
        "any_phantom_hp_claims": False,
        "turns_scored": 1,
    }


def test_score_phantom_hp_claims_ignores_ordinary_narration_with_no_hp_claim():
    """A tool-free turn that never claims a hit-point figure is not a phantom claim."""
    tool_log = [[], [], ["campaign_status"]]
    posts = [
        "You look around the market.",
        "I cannot resolve this action because the oar-case is not currently recorded.",
        "You check your inventory.",
    ]

    result = score_phantom_hp_claims(tool_log, posts)

    assert result["phantom_hp_turns"] == []
    assert result["any_phantom_hp_claims"] is False


def test_score_phantom_hp_claims_is_case_tolerant_and_catches_the_word_before_digit_shape():
    """``HP_FIGURE_PATTERN`` catches both \"<N> hit points\" and \"hit points ... <N>\".
    """
    tool_log = [[], []]
    posts = [
        "ossa is at 5 HP now.",
        "Rill checks the wound. Hit points are still 9, thankfully.",
    ]

    result = score_phantom_hp_claims(tool_log, posts)

    assert result["phantom_hp_turns"] == [1, 2]


def test_score_phantom_hp_claims_does_not_flag_a_character_sheet_lookup():
    """`character_sheet` is a read-only lookup, never an "hp-changing" tool by name,

    but its own envelope returns a real, current hit-point figure
    (``src/bsh_mcp/service.py``'s ``summary``/``sheet.hp``) for the narrator to
    state truthfully with no mutation at all. Before this tool's inclusion in
    ``HP_BACKING_TOOLS``, a live turn that called it and then honestly reported the
    real figure it returned would false-positive as a phantom claim -- the same
    regression class M19's own repair round found and fixed for
    `grant_runic_weapon`, checked here rather than only reasoned about.
    """
    tool_log = [["character_sheet"]]
    posts = ["Ossa: 12/15 hit points, Doom d10."]

    result = score_phantom_hp_claims(tool_log, posts)

    assert result["phantom_hp_turns"] == []
    assert result["any_phantom_hp_claims"] is False


def test_score_phantom_hp_claims_flags_a_combat_narration_with_no_backing_call():
    """A fabricated hit-point figure outside the rest-declaration shape is still caught.
    """
    tool_log = [[]]
    posts = [
        "The blade finds its mark. Rill takes 4 damage and drops to 9 hit points."
    ]

    result = score_phantom_hp_claims(tool_log, posts)

    assert result["phantom_hp_turns"] == [1]


def test_score_phantom_hp_claims_uses_the_engine_turn_posts_alignment_contract():
    """The same misattribution hazard ``score_phantom_roll_claims`` guards against."""
    config = _fake_config()
    tool_log = [["rest"], []]
    raw_posted = [
        "Ossa recovers 5 hit points (12/15).",
        config.risk_confirmation_notice,
        "Rill is now at 9 hit points after the scuffle.",
    ]

    misaligned = score_phantom_hp_claims(tool_log, raw_posted)
    # Raw indexing checks tool_log[1] (empty -- no HP-backing tool) against
    # raw_posted[1] (the notice, which does not match the pattern), so the real
    # unbacked claim at raw_posted[2] is never reached at all.
    assert misaligned["phantom_hp_turns"] == []

    post_log = [
        _turn_narration(raw_posted[0]),
        _service_notice(raw_posted[1], "risk_confirmation"),
        _turn_narration(raw_posted[2]),
    ]
    aligned = score_phantom_hp_claims(tool_log, engine_turn_posts(post_log))
    assert aligned["phantom_hp_turns"] == [2]


def test_rest_declaration_turns_matches_the_fixtures_default_six_turn_shape():
    """Rest declaration turns matches the fixtures default six turn shape."""
    assert rest_declaration_turns(42, {41}) == [5, 11, 17, 23, 29, 35]
    # With nothing claimed, 42 mentions alone would schedule a seventh rest turn at
    # 41 -- the general 6-mention-cycle rule below, unfiltered by any probe.
    assert rest_declaration_turns(42) == [5, 11, 17, 23, 29, 35, 41]


def test_rest_declaration_turns_extends_generally_past_the_defaults_six_turn_run():
    """A longer or shorter run still lands on every turn congruent to 5 mod 6.

    The milestone's own scope asks this to generalise past the fixture's default 42-
    turn shape rather than hard-code six positions.
    """
    assert rest_declaration_turns(48) == [5, 11, 17, 23, 29, 35, 41, 47]
    assert rest_declaration_turns(10) == [5]
    assert rest_declaration_turns(4) == []


def test_rest_declaration_turns_excludes_a_turn_a_different_probe_has_claimed():
    """A probe mention on a would-be rest turn is not a rest declaration.

    ``generate_transcript`` replaces the default six-shape rotation's mention on a
    probe's own claimed turn with that probe's own text, so a caller must exclude
    it -- otherwise a caller scoring, say, a session whose zone-A recall lands on
    turn 41 would misread that recall reply as a silent rest declaration.
    """
    assert rest_declaration_turns(42, {41}) == [5, 11, 17, 23, 29, 35]
    assert rest_declaration_turns(42, {5, 23}) == [11, 17, 29, 35, 41]


def test_score_rest_handling_detects_a_refused_rest():
    """Score rest handling detects a refused rest."""
    raw_posted = [
        'A short rest requires authenticated confirmation in a safe environment. I cannot resolve it yet.\n\nThe group waits beside a wall.'
    ]
    tool_log = [[]]

    result = score_rest_handling(raw_posted, [1], tool_log, raw_posted)

    assert result["rest_declarations"] == [{"turn": 1, "route": "refused"}]
    assert result["rest_tool_requested_count"] == 0


def test_score_rest_handling_detects_unbacked_totals():
    """Score rest handling detects unbacked totals."""
    raw_posted = [
        'Nobody enters the courtyard.\n\nRill, your hit points are still 9/9. Ossa, your hit points are 10/10.\n\nThe group waits.'
    ]
    tool_log = [[]]

    result = score_rest_handling(raw_posted, [1], tool_log, raw_posted)

    assert result["rest_declarations"] == [{"turn": 1, "route": "fabricated"}]


def test_score_rest_handling_detects_requested_recovery():
    """The one real, tool-backed rest declaration in the entire 13-session sample.
    """
    raw_posted = [
        'Ossa recovers 0 hit points (11/11).\nRill recovers 0 hit points (13/13).\n\nThe group waits beside a wall.'
    ]
    tool_log = [["rest"]]

    result = score_rest_handling(raw_posted, [1], tool_log, raw_posted)

    assert result["rest_declarations"] == [{"turn": 1, "route": "tool_requested"}]
    assert result["rest_tool_requested_count"] == 1
    assert result["rest_tool_requested_rate"] == 1.0


def test_score_rest_handling_tags_atmospheric_narration_with_no_call_or_figure_as_silent():
    """Test score rest handling tags atmospheric narration with no call or figure as silent.
    """
    raw_posted = [
        'The group pauses beside a wall. Clouds pass over an empty courtyard, and a distant shutter rattles.'
    ]
    tool_log = [[]]

    result = score_rest_handling(raw_posted, [1], tool_log, raw_posted)

    assert result["rest_declarations"] == [{"turn": 1, "route": "silent"}]


def test_score_rest_handling_aggregates_synthetic_samples():
    """Test score rest handling aggregates synthetic samples.
    """
    refused = "I cannot resolve the short rest because a safe environment is required."
    fabricated = "Rill, your hit points are still 9/9."
    silent = "You settle in for a quiet hour. Nothing approaches."
    tool_requested = 'Ossa recovers 0 hit points (11/11).\nRill recovers 0 hit points (13/13).\n\nThe group waits beside a wall.'

    raw_posted = [refused] * 4 + [fabricated] * 4 + [silent] * 4 + [tool_requested]
    tool_log = [[]] * 4 + [[]] * 4 + [[]] * 4 + [["rest"]]
    rest_turns = list(range(1, 14))

    result = score_rest_handling(raw_posted, rest_turns, tool_log, raw_posted)

    assert result["rest_declaration_count"] == 13
    assert result["rest_tool_requested_count"] == 1
    routes = [entry["route"] for entry in result["rest_declarations"]]
    assert routes.count("refused") == 4
    assert routes.count("fabricated") == 4
    assert routes.count("silent") == 4
    assert routes.count("tool_requested") == 1


def test_score_rest_handling_uses_the_engine_turn_posts_alignment_contract():
    """The same misattribution hazard ``score_phantom_roll_claims`` guards against.

    A pre-engine notice diverts one mention with no ``tool_log`` entry; matching
    ``tool_log`` positionally against raw turn order without first aligning through
    ``engine_turn_posts`` would check the wrong tool-log entry against the real rest
    declaration that follows the notice.
    """
    config = _fake_config()
    tool_log = [[], ["rest"]]
    raw_posted = [
        "I cannot resolve the short rest because a safe environment is required.",
        config.risk_confirmation_notice,
        "Ossa recovers 2 hit points (9/9).",
    ]

    post_log = [
        _turn_narration(raw_posted[0]),
        _service_notice(raw_posted[1], "risk_confirmation"),
        _turn_narration(raw_posted[2]),
    ]
    aligned = engine_turn_posts(post_log)
    result = score_rest_handling(raw_posted, [1, 3], tool_log, aligned)

    assert result["rest_declarations"] == [
        {"turn": 1, "route": "refused"},
        {"turn": 3, "route": "tool_requested"},
    ]
