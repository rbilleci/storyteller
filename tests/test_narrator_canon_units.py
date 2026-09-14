"""Test narrator canon units.
"""

from narrator_units_shared import *  # noqa: F401,F403 -- the split's shared header


def test_the_filter_strips_every_control_token_truncated_at_any_length():
    """The narrower prose rule must not cost the case the function exists for.

    ``max_tokens`` cuts at an arbitrary point, so every prefix of every name is a real
    truncation, not only the whole name. An audit found `` <start_of_turn`` reaching a
    player because the filter's name tuple held two entries while this file's family
    list held four.
    """
    for name in delivery._CONTROL_TOKEN_NAMES:
        for length in range(2, len(name) + 1):
            for cut in (f"<{name[:length]}", f"<|{name[:length]}"):
                cleaned, removed = delivery.scrub_markup(f"Narration. {cut}")
                assert cleaned == "Narration.", cut
                assert removed, cut


def test_present_npc_ids_strip_the_life_status_annotation():
    """``store.render_scene_markdown`` now annotates non-alive present NPCs
    ("- rade (dead)"); the digest's id parser must keep returning bare ids."""
    from narrator.canon import _present_npc_ids

    scene = (
        "## Present NPCs\n\n- rade (dead)\n- sera-vane\n- ammet (fled)\n\n"
        "## Open hooks\n\n- None recorded.\n"
    )
    assert _present_npc_ids(scene) == ["rade", "sera-vane", "ammet"]


def test_the_canon_digest_reads_three_sources_keyed_by_the_scene_render(tmp_path):
    """Scene render, current-location world file, present-NPC index entries."""
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: the-test-pier\nsession: 1\n---\n\n# Pier\n\n"
        "## Visible facts\n\n- The bell is cracked.\n\n"
        "## Present NPCs\n\n- test-keeper\n\n## Open hooks\n\n- None recorded.\n",
        encoding="utf-8",
    )
    (tmp_path / "world" / "locations").mkdir(parents=True)
    (tmp_path / "world" / "locations" / "the-test-pier.md").write_text(
        "## Public description\n\nA pier.\n\n## Hidden truths\n\nA snare under the third plank.\n",
        encoding="utf-8",
    )
    (tmp_path / "world" / "npcs").mkdir()
    (tmp_path / "world" / "npcs" / "index.yaml").write_text(
        "npcs:\n  - id: test-keeper\n    name: Keeper\n    motive: watch the pier\n"
        "  - id: absent-one\n    name: Absent\n    motive: never appear\n",
        encoding="utf-8",
    )

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200)

    assert digest.stats["location_id"] == "the-test-pier"
    assert digest.stats["present_npcs"] == ["test-keeper"]
    assert "A snare under the third plank." in digest.text
    assert "the world's" in digest.text and "reveal them only when play" in digest.text
    assert "motive: watch the pier" in digest.text
    assert "absent-one" not in digest.text
    assert digest.stats["truncated"] == {
        "scene": False, "location": False, "npcs": False, "exits": False, "resources": False,
    }


def test_the_canon_digest_renders_party_resources_from_character_sheets(tmp_path):
    """Test the canon digest renders party resources from character sheets.
    """
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: the-test-pier\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    _write_character(
        tmp_path, "rill", name="Rill", coins=12, hp=7, hp_max=9,
        weapons=["hunting bow"],
        equipment=["rope", "dagger"],
        resources=[{"id": "rations", "name": "Rations", "die": "d4"}],
    )
    _write_character(tmp_path, "ossa", name="Ossa", status="helpless", coins=3, hp=0, hp_max=10)

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200)

    assert "Party resources" in digest.text
    assert (
        "Rill (ok): 12 coins, HP 7/9, weapons: hunting bow, "
        "equipment: rope, dagger, resources: rations (d4)"
    ) in digest.text
    assert (
        "Ossa (helpless): 3 coins, HP 0/10, weapons: none, "
        "equipment: none, resources: none"
    ) in digest.text
    assert digest.stats["resources_chars"] > 0
    assert digest.stats["truncated"]["resources"] is False


def test_the_canon_digest_resources_block_is_bounded_and_counts_truncation(tmp_path):
    """A large party still gets a hard character bound, exactly like every other block."""
    from narrator.canon import TRUNCATION_MARKER, render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    for index in range(20):
        _write_character(
            tmp_path, f"pc-{index:02d}", name=f"Character {index:02d}",
            coins=index, hp=5, hp_max=10, equipment=["a long name of gear " * 3],
        )

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200, 200)

    assert digest.stats["truncated"]["resources"] is True
    assert digest.stats["resources_chars"] <= 200
    assert TRUNCATION_MARKER.strip() in digest.text


def test_the_canon_digest_resources_block_never_truncates_mid_digit(tmp_path):
    """BLOCKER-3 regression (independent audit, result-1.json).

    ``_bounded``'s raw byte-position cut could land inside a coin figure's own
    digits, showing a wrong partial number ("1234" sliced out of "1234567")
    immediately before the truncation marker -- worse than showing nothing, for a
    block whose whole instruction to the model is "copy this figure exactly". The
    resources block must drop the last partial line entirely instead.
    """
    from narrator.canon import TRUNCATION_MARKER, render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    for index in range(10):
        _write_character(
            tmp_path, f"pc-{index:02d}", name=f"Character {index:02d}",
            coins=1234567, hp=5, hp_max=10,
        )

    # Each full line runs about 80 characters; 300 keeps a few whole lines while
    # still cutting the ten-character party short -- and lands inside the digits
    # of "1234567 coins" on the boundary line under raw byte-position truncation.
    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200, 300)

    assert digest.stats["truncated"]["resources"] is True
    block = digest.text.split("current totals)\n\n", 1)[1]
    body = block.split(TRUNCATION_MARKER.strip(), 1)[0]
    # Every retained line is a complete, unmutilated source line -- reconstructed
    # here exactly as ``_character_summary`` composes it -- never a prefix that
    # cuts through the middle of a word or a digit run.
    retained_lines = [line for line in body.splitlines() if line.strip()]
    assert retained_lines, "the bound left zero whole lines, which the test cannot pin"
    for line in retained_lines:
        assert re.fullmatch(
            r"- Character \d{2} \(ok\): 1234567 coins, HP 5/10, weapons: none, "
            r"equipment: none, resources: none",
            line,
        ), f"a partial or mutilated line reached the digest: {line!r}"
    # No digit run in the retained body sits directly against the truncation
    # marker without a line boundary between them -- the concrete shape the
    # auditor's "1234" example named.
    assert not re.search(r"\d\n?$", body.rstrip("\n"))


def test_the_canon_digest_resources_read_fails_open_on_a_malformed_character_file(tmp_path):
    """One corrupt character file must not blank the digest or raise."""
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    _write_character(tmp_path, "rill", name="Rill", coins=4, hp=8, hp_max=9)
    characters = tmp_path / "campaign" / "characters"
    (characters / "broken.json").write_text("{not valid json", encoding="utf-8")

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200)

    assert "Rill (ok): 4 coins, HP 8/9" in digest.text
    assert digest.text.count("- ") >= 1


def test_the_canon_digest_resources_block_renders_the_same_in_the_public_scope(tmp_path):
    """A party's own coins and HP are not a world secret; the planner reads them too."""
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    _write_character(tmp_path, "rill", name="Rill", coins=4, hp=8, hp_max=9)

    public = render_digest(tmp_path, tmp_path, 2000, 2600, 1200, public=True).text
    assert (
        "Rill (ok): 4 coins, HP 8/9, weapons: none, equipment: none, resources: none"
    ) in public


def test_the_canon_digest_hp_clause_defers_to_the_tool(tmp_path):
    """Test the canon digest hp clause defers to the tool.
    """
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    _write_character(tmp_path, "rill", name="Rill", coins=4, hp=8, hp_max=9)

    text = render_digest(tmp_path, tmp_path, 2000, 2600, 1200).text

    assert "State a coin, item, or hit-point figure only when a tool result this turn returned it" in text
    assert "`campaign_status`" in text
    assert "`character_sheet`" in text
    assert "never licenses stating one" in text


    assert "hit-point figure you state must copy it exactly" not in text
    assert "any such figure you state must copy it exactly" not in text
    assert "A coin or item figure you state must copy it exactly" not in text


def test_the_canon_digest_coin_and_item_clause_defers_to_the_tool(tmp_path):
    """Coins and items now get the same tool-deferred treatment hit points already do.
    """
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    _write_character(tmp_path, "rill", name="Rill", coins=4, hp=8, hp_max=9)

    text = render_digest(tmp_path, tmp_path, 2000, 2600, 1200).text

    assert (
        "A coin, item, or hit-point figure is never estimated, recalled, or computed"
        in text
    )
    assert "A coin or item figure you state must copy it exactly" not in text
    assert "State a coin, item, or hit-point figure only when a tool result this turn returned it" in text


def test_the_canon_digest_coin_and_item_clause_would_have_failed_before_the_fix():
    """The same assertions, run against the pre-fix instruction text (the one
    M22 itself shipped, with the coin/item "must copy it exactly" clause still in
    place), must fail.
    """
    pre_fix = (
        "A coin, hit-point, or item figure is never estimated, recalled, or "
        "computed: the Party resources block below is the character sheet's own "
        "current numbers. A coin or item figure you state must copy it exactly. "
        "A hit-point figure is different, because a tool returns one: this block "
        "tells you where the party stands, and never licenses stating a number. "
        "State a hit-point figure only when a tool result this turn returned it, "
        "and call `campaign_status` or `character_sheet` when you want to state "
        "one and no tool has.\n\n"
    )
    assert "A coin or item figure you state must copy it exactly" in pre_fix
    assert (
        "State a coin, item, or hit-point figure only when a tool result this turn returned it"
        not in pre_fix
    )


def test_the_canon_digest_hp_clause_excepts_a_pending_rest_declaration(tmp_path):
    """Test the canon digest hp clause excepts a pending rest declaration.
    """
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    _write_character(tmp_path, "rill", name="Rill", coins=4, hp=8, hp_max=9)

    text = render_digest(tmp_path, tmp_path, 2000, 2600, 1200).text

    assert "except for a pending short- or long-rest declaration" in text
    assert "only `rest` supplies the figure for what the rest restored" in text
    assert "`campaign_status` and `character_sheet` return the pre-rest total" in text
    # The general clause itself, pinned by the earlier test in this round, is
    # unchanged -- this is an explicit exception added to it, not a replacement.
    assert "State a coin, item, or hit-point figure only when a tool result this turn returned it" in text


def test_the_canon_digest_hp_clause_rest_exception_would_have_failed_before_the_repair():
    """The same assertions, run against the exact pre-repair instruction text (commit
    f30a745f8c5d97da7e27f999f8cba1fd836e446e, the frozen candidate an independent
    systemic-assurance review found this gap in), must fail.
    """
    pre_repair = (
        "A coin, hit-point, or item figure is never estimated, recalled, or "
        "computed: the Party resources block below is the character sheet's own "
        "current numbers. A coin or item figure you state must copy it exactly. "
        "A hit-point figure is different, because a tool returns one: this block "
        "tells you where the party stands, and never licenses stating a number. "
        "State a hit-point figure only when a tool result this turn returned it, "
        "and call `campaign_status` or `character_sheet` when you want to state "
        "one and no tool has.\n\n"
    )
    assert "except for a pending short- or long-rest declaration" not in pre_repair
    assert "only `rest` supplies the figure for what the rest restored" not in pre_repair
    # The unconditional fallback the finding named is present, unexcepted, pre-repair.
    assert (
        "call `campaign_status` or `character_sheet` when you want to state "
        "one and no tool has.\n\n"
    ) in pre_repair


def test_the_canon_digest_hp_clause_would_have_failed_before_the_fix():
    """The new tests' own assertions, run against the exact pre-fix instruction text,
    must fail -- a guard never seen failing asserts nothing.
    """
    pre_fix = (
        "A coin, hit-point, or item figure is never estimated, recalled, or "
        "computed: the Party resources block below is the character sheet's own "
        "current numbers, and any such figure you state must copy it exactly.\n\n"
    )
    assert "State a hit-point figure only when a tool result this turn returned it" not in pre_fix
    assert "never licenses stating a number" not in pre_fix
    assert "any such figure you state must copy it exactly" in pre_fix


def test_the_canon_digest_bounds_every_block_and_counts_truncation(tmp_path):
    from narrator.canon import TRUNCATION_MARKER, render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: big\n---\n" + "F" * 5000, encoding="utf-8"
    )
    (tmp_path / "world" / "locations").mkdir(parents=True)
    (tmp_path / "world" / "locations" / "big.md").write_text("L" * 5000, encoding="utf-8")

    digest = render_digest(tmp_path, tmp_path, 400, 400, 400)

    assert digest.stats["truncated"]["scene"] is True
    assert digest.stats["truncated"]["location"] is True
    assert digest.stats["scene_chars"] == 400
    assert digest.stats["location_chars"] == 400
    assert digest.text.count(TRUNCATION_MARKER.strip()) == 2


def test_an_unreadable_canon_source_yields_an_empty_block_never_an_exception(tmp_path):
    """Fail open for context, closed for canon: reading less is the status quo."""
    from narrator.canon import render_digest

    digest = render_digest(tmp_path / "missing", tmp_path / "missing", 2000, 2600, 1200)
    assert digest.text == ""
    assert digest.stats["total_chars"] == 0


def test_an_undecodable_canon_source_yields_an_empty_block_never_an_exception(tmp_path):
    """One invalid byte in a human-edited world file must not end the session loop."""
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: the-test-pier\n---\n\n## Present NPCs\n- None recorded.\n",
        encoding="utf-8",
    )
    (tmp_path / "world" / "locations").mkdir(parents=True)
    (tmp_path / "world" / "locations" / "the-test-pier.md").write_bytes(
        b"## Hidden truths\n\xff\xfe not utf-8"
    )

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200)

    assert digest.stats["location_chars"] == 0
    assert "Authored location canon" not in digest.text
    assert "Current scene record" in digest.text


def test_head_retention_alone_would_take_the_clocks_section(tmp_path):
    """The defect this composition exists to fix, pinned against the retained geometry.
    """
    from narrator.canon import _bounded, _scene_sections

    text = _scene_render(19).strip()
    assert len(text) == 2496
    block, cut, kept = _bounded(text, 2000)
    states = {row["name"]: row["state"] for row in _scene_sections(text, kept)}

    assert cut is True
    assert states["Visible facts"] == "full"
    assert states["Objects"] == "partial"
    assert states["Clocks"] == "absent"
    assert states["Game-master-only facts"] == "absent"

    # One rung deeper, head retention reaches the fact list itself and takes every
    # section under it. This is the geometry the 60-fact golden encodes.
    deep = _scene_render(60).strip()
    _, deep_cut, deep_kept = _bounded(deep, 2000)
    deep_states = {row["name"]: row for row in _scene_sections(deep, deep_kept)}

    assert deep_cut is True
    assert deep_states["Visible facts"]["state"] == "partial"
    assert 0 < deep_states["Visible facts"]["items_retained"] < 60
    for name in (
        "Exits", "Objects", "Present NPCs", "Open hooks", "Clocks", "Unratified outcomes",
    ):
        assert deep_states[name]["state"] == "absent", name


def test_composition_keeps_every_protected_section_at_the_retained_geometry(tmp_path):
    """The same record, composed by value: the protected sections all survive."""
    from narrator.canon import _PROTECTED_SECTIONS, render_digest

    digest = render_digest(_canon_root(tmp_path, 19), tmp_path, 2000, 2600, 1200)

    assert digest.stats["truncated"]["scene"] is True
    assert digest.stats["scene_policy"] == "priority"
    states = _states(digest)
    # ``Combat`` is the one protected name this fixture omits, because it runs outside
    # combat and the renderer emits the section only while a fight runs. Pinning the
    # exact absent set keeps a future disappearance loud instead of silently skipped.
    assert {n for n in _PROTECTED_SECTIONS if n not in states} == {"Combat"}
    for name in _PROTECTED_SECTIONS:
        if name == "Combat":
            continue
        assert states[name] == "full", name
    assert digest.stats["scene_chars"] <= 2000


def test_scene_section_survival_reports_every_section_full_when_the_block_fits(tmp_path):
    """A run under the bound must record zero loss, so an absent row means a real cut."""
    from narrator.canon import render_digest

    digest = render_digest(_canon_root(tmp_path, 4), tmp_path, 2000, 2600, 1200)

    assert digest.stats["truncated"]["scene"] is False
    assert digest.stats["scene_policy"] == "whole"
    assert set(_states(digest).values()) == {"full"}
    facts = next(
        section
        for section in digest.stats["scene_sections"]
        if section["name"] == "Visible facts"
    )
    assert facts["items"] == 4 and facts["items_retained"] == 4


def test_the_fact_list_absorbs_the_cut_while_the_protected_sections_hold(tmp_path):
    """The trade this slice makes, stated in one assertion pair.

    The fact list keeps head retention, so its oldest entries stay and its tail
    absorbs the cut. The protected sections stay resident at a depth where head
    retention alone would have taken every one of them.
    """
    from narrator.canon import _PROTECTED_SECTIONS, render_digest

    digest = render_digest(_canon_root(tmp_path, 60), tmp_path, 2000, 2600, 1200)
    sections = digest.stats["scene_sections"]
    facts = next(s for s in sections if s["name"] == "Visible facts")
    states = {s["name"]: s["state"] for s in sections}

    assert facts["state"] == "partial"
    assert 0 < facts["items_retained"] < facts["items"] == 60
    # ``Combat`` is the one protected name this fixture omits, because it runs outside
    # combat and the renderer emits the section only while a fight runs. Pinning the
    # exact absent set keeps a future disappearance loud instead of silently skipped.
    assert {n for n in _PROTECTED_SECTIONS if n not in states} == {"Combat"}
    for name in _PROTECTED_SECTIONS:
        if name == "Combat":
            continue
        assert states[name] == "full", name
    assert states["Game-master-only facts"] == "absent"
    assert digest.stats["scene_chars"] <= 2000


def test_the_kept_facts_are_the_oldest_because_head_retention_holds(tmp_path):
    """Newest-first or both-ends selection would evict the digest-only middle.
    """
    from narrator.canon import render_digest

    digest = render_digest(_canon_root(tmp_path, 60), tmp_path, 2000, 2600, 1200)

    assert "- Fact 01:" in digest.text
    assert "- Fact 60:" not in digest.text


def test_the_typed_object_map_is_digest_resident_at_the_deepest_cut(tmp_path):
    """Test the typed object map is digest resident at the deepest cut.
    """
    from narrator.canon import render_digest

    digest = render_digest(_canon_root(tmp_path, 60), tmp_path, 2000, 2600, 1200)

    assert "## Objects" in digest.text
    assert "- tower-door (locked) an iron slide bolt" in digest.text
    section = _states(digest)
    assert section["Objects"] == "full"


def test_protected_sections_larger_than_the_budget_fall_back_to_head_retention():
    """Fail open on the degenerate record too: a bound smaller than the resident set.

    Reserving more than the budget would leave the fact list a negative allowance, so
    the composer declines and returns what head retention returns.
    """
    from narrator.canon import _compose_scene_block

    text = (
        "---\nlocation_id: x\n---\n\n## Summary\n\n" + "S" * 3000
        + "\n\n## Visible facts\n\n- a\n\n## Clocks\n\n- c (1/6)\n"
    ).strip()

    block, cut, rows, policy = _compose_scene_block(text, 2000)

    assert policy == "head"
    assert cut is True
    assert len(block) == 2000


def test_an_unrecognized_render_falls_back_to_head_retention(tmp_path):
    """Fail open: a render this module cannot parse must not lose more than today."""
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir(parents=True)
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: nowhere\n---\n\n" + "prose without any section heading. " * 200,
        encoding="utf-8",
    )

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200)

    assert digest.stats["scene_policy"] == "head"
    assert digest.stats["truncated"]["scene"] is True
    assert digest.stats["scene_chars"] == 2000


def test_a_non_truncating_synthetic_digest_has_stable_bytes(tmp_path):
    """Pin the synthetic scene, including its fixed year-2000 timestamp."""
    import hashlib

    from narrator.canon import render_digest

    root = _canon_root(tmp_path / "fits", 4)
    digest = render_digest(root, root, 2000, 2600, 1200)

    assert digest.stats["truncated"]["scene"] is False
    assert (
        hashlib.sha256(digest.text.encode("utf-8")).hexdigest()
        == "950619c19ba652dc0c1490c4762a3832b4e18dda5a52760cc74570643b989d54"
    )


def test_truncating_synthetic_digests_preserve_objects_within_the_bound(tmp_path):
    """Priority rendering keeps the object section when facts exceed the budget."""
    from narrator.canon import render_digest

    for fact_count in (19, 60):
        root = _canon_root(tmp_path / str(fact_count), fact_count)
        digest = render_digest(root, root, 2000, 2600, 1200)
        assert digest.stats["scene_policy"] == "priority"
        assert digest.stats["truncated"]["scene"] is True
        assert digest.stats["scene_chars"] <= 2000
        assert "## Objects" in digest.text


def test_the_payload_measurement_supersedes_every_canon_block_but_the_newest():
    """The current digest renders the same three sources at their latest state, so every
    earlier copy in the same request is superseded by construction. The measurement
    therefore counts the last block as current and the rest as reclaimable.
    """
    from narrator.payload import measure_request

    turns = [_digest_turn(f"block {index}") for index in range(3)]
    row = measure_request(_request(*[text for text, _ in turns]))

    assert row["canon_blocks"] == 3
    assert row["canon_current_chars"] == len(turns[-1][1]) + 2
    assert row["canon_superseded_chars"] == sum(
        len(digest) + 2 for _, digest in turns[:-1]
    )


def test_the_measured_canon_span_is_the_span_turn_prompt_wrote():
    """The two markers bound the digest exactly, so no channel text enters the count."""
    from narrator.payload import measure_request

    text, digest = _digest_turn("one block", channel="Rill: the tide is turning")
    row = measure_request(_request(text))

    assert row["canon_current_chars"] == len(digest) + len("\n\n")
    assert row["canon_superseded_chars"] == 0
    assert row["messages_chars"] == len(text)


def test_a_request_carrying_no_canon_reports_nothing_to_reclaim():
    """A settle or sweep request carries one prompt and no digest at all."""
    from narrator.payload import measure_request

    row = measure_request(_request("Settle this ledger entry."))

    assert row["canon_blocks"] == 0
    assert row["canon_current_chars"] == 0
    assert row["canon_superseded_chars"] == 0


def test_stripping_leaves_the_prompt_the_pre_digest_runtime_sent():
    """Every resident digest goes, and the channel text stays byte for byte.

    The engine strips before it appends the current turn, so every digest present is
    superseded. A stripped message equals the prompt this project sent before the
    digest existed, which makes the removal's result inspectable rather than novel.
    """
    from narrator.payload import strip_superseded_canon

    channels = ["Rill: one", "Ossa: two", "Rill: three"]
    messages = _history(*[_digest_turn("block", channel)[0] for channel in channels])

    assert strip_superseded_canon(messages) == 3
    assert [message["content"][0]["text"] for message in messages] == [
        prompt.turn_prompt(channel) for channel in channels
    ]


def test_stripping_leaves_a_message_carrying_no_digest_untouched():
    """A settle prompt, a bare turn, and a hand-built message keep every byte."""
    from narrator.payload import strip_superseded_canon

    texts = [prompt.turn_prompt("Rill: hello"), "Settle this ledger.", ""]
    messages = _history(*texts)

    assert strip_superseded_canon(messages) == 0
    assert [message["content"][0]["text"] for message in messages] == texts


def test_a_record_quoting_the_body_marker_costs_canon_and_never_player_text():
    """The cut is conservative in one direction, and this is that direction.

    A scene record holding the channel body's own marker moves the cut earlier, so a
    digest tail survives. The alternative — searching from the end — would delete
    player text whenever a player typed the marker, and player text is the one thing
    the conversation window alone carries.
    """
    from narrator.canon import DIGEST_HEADING
    from narrator.payload import strip_superseded_canon

    hostile = f"{DIGEST_HEADING}\n\n- A note reading {prompt.TURN_BODY_MARKER}here."
    messages = _history(prompt.turn_prompt("Rill: the note is a forgery", canon=hostile))

    original = messages[0]["content"][0]["text"]
    assert strip_superseded_canon(messages) == 1
    kept = messages[0]["content"][0]["text"]
    assert kept.endswith(prompt.turn_prompt("Rill: the note is a forgery"))
    assert len(kept) < len(original)
    # The residue is the record's own tail, cut at the marker it quoted.
    assert kept.startswith(f"{prompt.TURN_BODY_MARKER}here.")


def test_the_digest_heading_constant_is_the_line_the_digest_actually_opens_with(tmp_path):
    """A drifted constant would silently measure zero canon in every request."""
    from narrator.canon import DIGEST_HEADING, render_digest

    root = _canon_root(tmp_path, 4)
    assert prompt.TURN_BODY_MARKER in prompt.turn_prompt("Rill: hi")
    assert render_digest(root, root, 2000, 2600, 1200).text.startswith(DIGEST_HEADING)


def test_turn_prompt_prepends_canon_and_is_byte_identical_without_it():
    with_canon = prompt.turn_prompt("Rill: hello", canon="# Scene canon\n\nfacts")
    without = prompt.turn_prompt("Rill: hello")
    assert with_canon.startswith("# Scene canon")
    assert with_canon.endswith(without)
    assert without == (
        "Discussion since your previous reply:\n\nRill: hello\n\nResolve this turn."
    )


def test_a_closed_fight_spends_no_digest_characters(tmp_path):
    """The section costs zero for every turn outside combat, which is most turns."""
    from narrator.canon import render_digest

    without = render_digest(_canon_root(tmp_path / "a", 4), tmp_path / "a", 8000, 2600, 1200)
    with_fight = render_digest(
        _canon_root(tmp_path / "b", 4, combat=True), tmp_path / "b", 8000, 2600, 1200
    )
    assert "## Combat" not in without.text
    assert "## Combat" in with_fight.text
    assert with_fight.stats["scene_chars"] > without.stats["scene_chars"]


def test_the_public_digest_strips_every_hidden_location_and_npc_field(tmp_path):
    from narrator.canon import render_digest

    public = render_digest(_disclosure_root(tmp_path), tmp_path, 8000, 2600, 1200, public=True).text

    # Player-perceivable content survives.
    assert "Stalls on the mud at low water." in public
    assert "The planks are slick." in public
    assert "Rope and salt for sale." in public
    assert "Sera Vane" in public
    # Secrets never reach the planner, whose output becomes player-visible text.
    assert "The keeper sold the writ's twin." not in public
    assert "Sera Vane wants the ledger back." not in public
    assert "bent brass clapper" not in public
    assert "Choir Below drowns" not in public
    assert "the-choir-below" not in public
    assert "bell-keeper of the road shrine" not in public
    assert "close the negotiation honestly" not in public
    assert "the-listener" not in public  # frontmatter hidden_entities
    # The scene's game-master-only tail is gone too.
    assert "Never quote this section to players" not in public
    assert "a hidden truth the players have not uncovered" not in public


def test_the_full_digest_still_carries_every_secret_for_the_narrator(tmp_path):
    """Regression: the narrator digest is unchanged; the delivery gate protects it."""
    from narrator.canon import render_digest

    full = render_digest(_disclosure_root(tmp_path), tmp_path, 8000, 2600, 1200).text

    assert "The keeper sold the writ's twin." in full
    assert "bell-keeper of the road shrine" in full
    assert "the-choir-below" in full
    assert "close the negotiation honestly" in full
    assert "Choir Below drowns" in full
    assert "Never quote this section to players" in full


def test_render_digest_carries_exit_display_names_beside_their_slugs(tmp_path):
    """End-to-end wiring: a fresh, self-contained fixture rather than ``_canon_root``,
    because several ``_canon_root``-based tests above pin ``render_digest``'s exact
    total character count, and adding ``world/locations/index.yaml`` there would grow
    every one of those digests by a new Exit canon block."""
    from narrator.canon import render_digest

    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "scene.md").write_text(
        "---\nlocation_id: the-road-shrine\nsession: 1\nin_game_minutes: 0\n"
        "updated_at: 2026-01-01T00:00:00Z\n---\n\n# The Road Shrine\n\n"
        "## Summary\n\nTest.\n\n## Visible facts\n\n- None recorded.\n\n"
        "## Exits\n\n- the-black-bell-crypt\n\n## Objects\n\n- None recorded.\n\n"
        "## Present NPCs\n\n- None recorded.\n",
        encoding="utf-8",
    )
    (tmp_path / "world" / "locations").mkdir(parents=True)
    (tmp_path / "world" / "locations" / "index.yaml").write_text(
        "locations:\n"
        "  - id: the-road-shrine\n    name: The Road Shrine\n"
        "    exits: [the-black-bell-crypt]\n"
        "  - id: the-black-bell-crypt\n    name: The Black Bell Crypt\n"
        "    exits: [the-road-shrine]\n",
        encoding="utf-8",
    )

    digest = render_digest(tmp_path, tmp_path, 2000, 2600, 1200, public=True)

    assert "## Exit canon" in digest.text
    assert "id: the-black-bell-crypt" in digest.text
    assert "name: The Black Bell Crypt" in digest.text
    assert digest.stats["exits"] == ["the-black-bell-crypt"]
    assert digest.stats["truncated"]["exits"] is False

    # A scene with no location-index file on disk (an older campaign, or a fixture
    # that predates this fix) renders no Exit canon block and raises nothing.
    no_index = tmp_path / "no-index"
    (no_index / "campaign").mkdir(parents=True)
    (no_index / "campaign" / "scene.md").write_text(
        (tmp_path / "campaign" / "scene.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    bare = render_digest(no_index, no_index, 2000, 2600, 1200, public=True)
    assert "## Exit canon" not in bare.text


def test_the_character_digest_states_the_weapons_a_character_carries(tmp_path):
    """Both authored characters carry an empty ``equipment`` list and a populated
    ``weapons`` list, so before this change every digest line about them read
    \"equipment: none\" for a party holding a dagger, a bow, and a knife.
    """
    from narrator.canon import _character_summary

    armed = _character_summary(json.dumps({
        "id": "ossa", "name": "Ossa", "status": "ok", "coins": 3, "hp": 9, "hp_max": 9,
        "weapons": ["duelling dagger"], "equipment": [], "resources": [],
    }))
    assert "weapons: duelling dagger" in armed
    assert "equipment: none" in armed
    # The defect in one sentence: the line must not read as carrying nothing at all.
    assert armed.count("none") == 2, armed

    for path in sorted((REPO_ROOT / "campaign" / "characters").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        line = _character_summary(json.dumps(payload))
        for weapon in payload.get("weapons") or []:
            assert weapon in line, (path.name, line)

    # An unarmed character, and a sheet predating the field, both still render.
    unarmed = _character_summary(json.dumps({
        "id": "scribe", "name": "Scribe", "hp": 4, "hp_max": 4, "weapons": [],
    }))
    assert "weapons: none" in unarmed
    legacy = _character_summary(json.dumps({"id": "old", "name": "Old", "hp": 1, "hp_max": 1}))
    assert "weapons: none" in legacy


def _fact_text(index: int) -> str:
    return (
        f"Fact {index:02d}: the party recorded a durable change at the eel market "
        "before slack tide."
    )


def _write_statements(root, statements, persons=None, npcs=None, objects=None) -> None:
    """A ``state.json`` in the shape ``_fact_priorities`` reads, beside the render."""
    state = {
        "scene": {
            "statements": statements,
            "persons": persons or {},
            "present_npcs": npcs or [],
            "objects": objects or {},
        }
    }
    (root / "campaign" / "state.json").write_text(json.dumps(state), encoding="utf-8")


def test_refs_selection_rescues_a_referenced_fact_beyond_the_head(tmp_path):
    """A cut fact about a present entity survives; the head's newest unref'd fact pays.

    The 60-fact fixture cuts deep into the fact list. Fact 55 sits far beyond the
    retained head, and its refs name a present scene person, so reference-keyed
    selection rescues it -- in original document order, funded by evicting the
    newest unreferenced fact the head would otherwise have kept. Retained counts
    match the baseline: the budget bought the same number of lines, differently.
    """
    from narrator.canon import render_digest

    root = _canon_root(tmp_path, 60)
    unreferenced = [{"text": _fact_text(i), "refs": [], "scope": "public"} for i in range(1, 61)]
    _write_statements(root, unreferenced)
    baseline = render_digest(root, tmp_path, 2000, 2600, 1200)

    statements = [{"text": _fact_text(i), "refs": [], "scope": "public"} for i in range(1, 61)]
    statements[54]["refs"] = ["orso-pell"]
    _write_statements(root, statements, persons={"orso-pell": {"name": "Orso Pell"}})
    digest = render_digest(root, tmp_path, 2000, 2600, 1200)

    assert "- Fact 55:" not in baseline.text
    assert "- Fact 55:" in digest.text
    assert "- Fact 01:" in digest.text  # the head still leads
    assert "- Fact 60:" not in digest.text  # unreferenced tail stays cut
    assert digest.stats["scene_policy"] == "refs"

    facts_row = next(
        s for s in digest.stats["scene_sections"] if s["name"] == "Visible facts"
    )
    baseline_row = next(
        s for s in baseline.stats["scene_sections"] if s["name"] == "Visible facts"
    )
    assert facts_row["items_rescued"] == 1
    assert facts_row["items_retained"] == baseline_row["items_retained"]
    # Same budget, same line lengths: the rescue displaced exactly the newest kept
    # unreferenced fact, and every other retained line is unchanged.
    evicted = baseline_row["items_retained"]
    assert f"- Fact {evicted:02d}:" in baseline.text
    assert f"- Fact {evicted:02d}:" not in digest.text


def test_refs_selection_is_byte_identical_when_no_referenced_fact_is_cut(tmp_path):
    """Refs inside the retained head change nothing: the legacy prefix, byte for byte."""
    from narrator.canon import render_digest

    root = _canon_root(tmp_path, 60)
    unreferenced = [{"text": _fact_text(i), "refs": [], "scope": "public"} for i in range(1, 61)]
    _write_statements(root, unreferenced)
    baseline = render_digest(root, tmp_path, 2000, 2600, 1200)

    statements = [{"text": _fact_text(i), "refs": [], "scope": "public"} for i in range(1, 61)]
    statements[0]["refs"] = ["orso-pell"]
    _write_statements(root, statements, persons={"orso-pell": {"name": "Orso Pell"}})
    digest = render_digest(root, tmp_path, 2000, 2600, 1200)

    assert digest.text == baseline.text
    assert digest.stats["scene_policy"] == "priority"


def test_refs_selection_ignores_a_referenced_fact_about_an_absent_entity(tmp_path):
    """Refs to nothing present rescue nothing: relevance is present-entity relevance."""
    from narrator.canon import render_digest

    root = _canon_root(tmp_path, 60)
    unreferenced = [{"text": _fact_text(i), "refs": [], "scope": "public"} for i in range(1, 61)]
    _write_statements(root, unreferenced)
    baseline = render_digest(root, tmp_path, 2000, 2600, 1200)

    statements = [{"text": _fact_text(i), "refs": [], "scope": "public"} for i in range(1, 61)]
    statements[54]["refs"] = ["a-departed-traveler"]
    _write_statements(root, statements, persons={"orso-pell": {"name": "Orso Pell"}})
    digest = render_digest(root, tmp_path, 2000, 2600, 1200)

    assert digest.text == baseline.text
    assert "- Fact 55:" not in digest.text


def test_refs_data_failures_fall_open_to_the_legacy_geometry(tmp_path):
    """An unreadable, legacy, or refs-free record keeps head retention byte for byte."""
    from narrator.canon import render_digest

    root = _canon_root(tmp_path, 60)
    baseline = render_digest(root, tmp_path, 2000, 2600, 1200)

    # No state.json at all (the fixture default).
    assert render_digest(root, tmp_path, 2000, 2600, 1200).text == baseline.text

    # Malformed state.json.
    (root / "campaign" / "state.json").write_text("{not json", encoding="utf-8")
    assert render_digest(root, tmp_path, 2000, 2600, 1200).text == baseline.text

    # A record whose statements carry no refs anywhere.
    statements = [{"text": _fact_text(i), "refs": [], "scope": "public"} for i in range(1, 61)]
    _write_statements(root, statements)
    digest = render_digest(root, tmp_path, 2000, 2600, 1200)
    assert digest.text == baseline.text
    assert digest.stats["scene_policy"] == "priority"


def test_a_fitting_block_is_untouched_by_refs_retrieval(tmp_path):
    """Under the bound nothing is selected because nothing is cut."""
    from narrator.canon import render_digest

    root = _canon_root(tmp_path, 5)
    statements = [{"text": _fact_text(i), "refs": ["orso-pell"], "scope": "public"} for i in range(1, 6)]
    _write_statements(root, statements, persons={"orso-pell": {"name": "Orso Pell"}})

    digest = render_digest(root, tmp_path, 8000, 2600, 1200)

    assert digest.stats["scene_policy"] == "whole"


def test_the_party_counts_as_present_for_refs_selection(tmp_path):
    """A fact referencing a party character is rescued: the party is always present."""
    from narrator.canon import render_digest

    root = _canon_root(tmp_path, 60)
    characters = root / "campaign" / "characters"
    characters.mkdir()
    (characters / "ossa.json").write_text(
        json.dumps({"id": "ossa", "name": "Ossa", "hp": 5, "hp_max": 5}), encoding="utf-8"
    )
    statements = [{"text": _fact_text(i), "refs": [], "scope": "public"} for i in range(1, 61)]
    statements[54]["refs"] = ["ossa"]
    _write_statements(root, statements)

    digest = render_digest(root, tmp_path, 2000, 2600, 1200)
    assert "- Fact 55:" in digest.text
    assert digest.stats["scene_policy"] == "refs"
