"""A miniature soak against the live endpoint, proving the compaction machinery works.

What this pins is the harness itself. If the Strands API renames
``removed_message_count``, if the service loop stops closing the adapter before the
engine, or if the generator drifts past the backfill cap, this test fails in the
acceptance suite rather than inside a later evidence run.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest

from narrator.soak_harness import bootstrap_sandbox, run_soak
from narrator.soak_instruments import PLAYERS, SoakReport, generate_transcript

MESSAGES = 30
BLOCK = 5
WINDOW = 6
SEED = 99


def test_the_generator_is_deterministic_and_two_player():
    """Same arguments, same bytes; both players speak; blocks respect the grammar."""
    first = generate_transcript(MESSAGES, BLOCK, SEED)
    second = generate_transcript(MESSAGES, BLOCK, SEED)
    assert first == second

    lines = [line for line in first.splitlines() if line and not line.startswith("#")]
    assert len(lines) == MESSAGES
    mentions = [line for line in lines if line.startswith("@GM")]
    assert len(mentions) == MESSAGES // BLOCK
    speakers = {line.split(":", 1)[0] for line in lines if not line.startswith("@GM")}
    assert speakers == set(PLAYERS)


def test_the_continuity_probe_is_deterministic_and_token_hygienic():
    """The probe's scoring integrity rests on three model-free properties.
    """
    from narrator.soak_instruments import (
        _CHATTER,
        _MENTIONS,
        AUTHORED_RECALL,
        AUTHORED_TOKENS,
        CONTINUITY_RECALL,
        CONTINUITY_TOKENS,
        normalize_for_lenient_match,
    )

    plain = generate_transcript(MESSAGES, BLOCK, SEED)
    assert plain == generate_transcript(MESSAGES, BLOCK, SEED, plant_turn=0, recall_turn=0)

    probed = generate_transcript(1050, 25, SEED, plant_turn=3, recall_turn=40)
    assert probed == generate_transcript(1050, 25, SEED, plant_turn=3, recall_turn=40)
    mentions = [line for line in probed.splitlines() if line.startswith("@GM")]
    assert all(token in mentions[2].lower() for token in CONTINUITY_TOKENS)
    assert not any(token in mentions[39].lower() for token in CONTINUITY_TOKENS)

    pools = (
        " ".join(_CHATTER)
        + " ".join(_MENTIONS)
        + CONTINUITY_RECALL
        + AUTHORED_RECALL
    ).lower()
    assert not any(token in pools for token in CONTINUITY_TOKENS + AUTHORED_TOKENS)

    # The lenient channel widens what counts as a token hit, so its hygiene must
    # widen too: a normalized token appearing in the normalized pools would let a
    # lenient pass come from echo rather than recall.
    normalized_pools = normalize_for_lenient_match(pools)
    assert not any(
        normalize_for_lenient_match(token) in normalized_pools
        for token in CONTINUITY_TOKENS + AUTHORED_TOKENS
    )


def test_lenient_normalization_tolerates_formatting_not_order():
    """The lenient channel forgives punctuation and case, never rearrangement."""
    from narrator.soak_instruments import normalize_for_lenient_match

    assert normalize_for_lenient_match("Salt–Cellar!") == "salt cellar"
    assert normalize_for_lenient_match("  tin\nwhistle  ") == "tin whistle"
    reply = normalize_for_lenient_match("She lifts the old salt-cellar, dented.")
    assert normalize_for_lenient_match("salt cellar") in reply
    assert normalize_for_lenient_match("cellar salt") not in reply
    twice = normalize_for_lenient_match(normalize_for_lenient_match("A—B c"))
    assert twice == normalize_for_lenient_match("A—B c")


def test_a_different_seed_changes_the_chatter_but_not_the_shape():
    other = generate_transcript(MESSAGES, BLOCK, SEED + 1)
    baseline = generate_transcript(MESSAGES, BLOCK, SEED)
    assert other != baseline
    assert len(other.splitlines()) == len(baseline.splitlines())


@pytest.mark.live
@pytest.mark.asyncio
async def test_a_short_soak_triggers_compaction_and_keeps_every_delivery_guarantee(tmp_path):
    """Thirty live player messages through the real service loop, window of six.

    Marked ``live``: this test reaches the served model over the network.
    ``package.json``'s ``check:unit:*`` narrator suite excludes ``-m live`` so an
    offline validate run never makes a GPU call, regardless of ambient environment.
    """
    if not os.environ.get("BSH_SERVER_PYTHON"):
        pytest.skip("BSH_SERVER_PYTHON must name an interpreter holding mcp 2.x")
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
    except (urllib.error.URLError, OSError):
        pytest.skip("the narrator endpoint is not serving")

    # The same bootstrap the evidence run uses: campaign files plus both characters,
    # created under the interpreter that holds mcp 2.x.
    bootstrap_sandbox(os.environ["BSH_SERVER_PYTHON"], tmp_path)

    transcript = tmp_path / "mini-soak.txt"
    transcript.write_text(generate_transcript(MESSAGES, BLOCK, SEED), encoding="utf-8")

    report = SoakReport(seed=SEED, window_size=WINDOW, generated_messages=MESSAGES)
    await run_soak(tmp_path, transcript, WINDOW, report)


    expected_failures = {"volume"}


    assert set(report.failed()) <= expected_failures, (report.checks, report.errors)

    assert report.fed_messages == MESSAGES
    assert report.turns == MESSAGES // BLOCK
    assert set(report.authors) == set(PLAYERS)
    assert report.errors == []
    assert report.removed_message_count > 0
    assert report.final_message_count <= WINDOW
    assert report.compaction_triggered

    # The dedup guarantee, pinned live rather than by the check list alone: a run that
    # stopped recording payload composition would drop the check silently, and
    # `set(report.failed())` would still hold. Six turns append six digests, so the
    # retention arm would reach six here.
    dedup = [c for c in report.checks if c["name"] == "one-canon-block-per-request"]
    assert len(dedup) == 1 and dedup[0]["pass"], report.checks
    assert report.payload["chars"]["canon_blocks_max"] == 1
    assert report.payload["chars"]["canon_superseded_chars_max"] == 0


    cache = report.cache
    assert cache is not None and cache["sampled"], cache
    assert cache["samples_taken"] >= 2
    assert cache["counters_missing"] == [], cache["counters_missing"]
    assert cache["window"]["prefix_cache_queries"] > 0, cache["window"]
    assert 0.0 <= cache["hit_rate"] <= 1.0, cache["hit_rate"]
    assert len(cache["per_turn"]) <= report.turns
    assert "usage_cache_key_present" in cache
    assert cache["engine_version"], cache

    # Per-request usage, pinned live because the pairing rests on an ordering fact:
    # NarratorOpenAIModel.format_request records a row, and the usage event for that
    # same request arrives before the stream call returns. A Strands change that moved
    # usage out of the stream, or that emitted it for a different request, would leave
    # rows unpaired here rather than in a later evidence run.
    per_request = report.payload["per_request_usage"]
    assert per_request["requests_with_usage"] > 0, per_request
    assert per_request["by_origin"]["turn"]["requests"] > 0, per_request
    assert per_request["input_tokens"] > 0
    assert 0 <= per_request["cached_tokens"] <= per_request["input_tokens"]
    for row in report.payload["per_request"]:
        usage = row.get("usage")
        if usage is None:
            continue
        assert usage["inputTokens"] > 0, row
        assert 0 <= usage.get("cacheReadInputTokens", 0) <= usage["inputTokens"], row
