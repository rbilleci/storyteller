"""Live rate measurement for the reasoning-leak finding from the session-close probe.

``tests_narrator/test_probe_session_close_debt.py`` found one live turn (of 20) whose
``outcome.narration`` was several paragraphs of visible chain-of-thought ("Wait, I
should check if I need to call `session_close` again... The user's prompt...") glued
onto a final sentence, delivered to the player as-is. Root-cause reading of
``src/narrator/engine.py``'s ``_assistant_text`` and the installed
``strands.models.openai.OpenAIModel`` (which correctly splits a streamed
``reasoning_content`` delta into its own ``reasoningContent``-typed content block,
excluded by ``_assistant_text``'s ``.get("text", "")`` join) found no extraction bug in
this repository's code: the only way that text reaches ``outcome.narration`` is if the
serving model never signalled it as reasoning in the first place -- a decode-time model
reliability variance, not a client-side parsing defect.

This measures how often it actually happens, and whether it clusters on ambiguous or
redundant turns (like the repro that first found it -- asking to close an
already-closed session) versus ordinary declarative turns, before any fix is designed.
No hard hallmark of "reasoning leaked" is provable without the raw un-parsed model
stream, which nothing here retains, so this uses conservative textual proxies (verbose
length, and phrases like "the user" or a backtick-quoted tool name that never belong in
first-person in-fiction prose) as flags for human review, and prints every flagged
narration in full rather than trusting the heuristic's count alone -- the same caution
that got ``leaked_tool_name`` deleted for over-triggering on "trestles" applies here:
this script never gates or scrubs, it only labels for the person reading the report.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.channels.base import ChannelMessage, InboundTurn  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402
from narrator.engine import NarratorEngine  # noqa: E402

_SUSPECT_PHRASES = (
    "the user", "session_close`", "`session_close", "i should call", "i need to check",
    "wait, i", "wait, looking", "actually, the", "let me think", "let me check",
)


def _suspect(narration: str) -> bool:
    lowered = narration.casefold()
    return len(narration) > 600 or any(phrase in lowered for phrase in _SUSPECT_PHRASES)


@pytest.fixture
def engine(tmp_path):
    if not os.environ.get("BSH_SERVER_PYTHON"):
        pytest.skip("BSH_SERVER_PYTHON must name an interpreter holding mcp 2.x")

    import shutil
    import subprocess

    for directory in ("rules", "world"):
        source = REPO_ROOT / directory
        if source.is_dir():
            shutil.copytree(source, tmp_path / directory)

    subprocess.run(
        [os.environ["BSH_SERVER_PYTHON"], str(REPO_ROOT / "scripts" / "new_campaign.py"),
         "--root", str(tmp_path), "--title", "Narrator test", "--seed-scene"],
        check=True, capture_output=True, cwd=str(REPO_ROOT),
    )

    config = NarratorConfig(campaign_root=tmp_path, repo_root=REPO_ROOT)
    engine = NarratorEngine(config)
    engine.start()
    yield engine
    engine.stop()


_ORDINARY_DECLARATIONS = (
    "Rill searches the market stalls for anything useful.",
    "Rill asks the fishmonger about the bell-keeper.",
    "Rill draws her long knife and watches the shadows.",
    "Rill offers to help carry the crates off the ferry.",
    "Rill hums a tune while walking the planks.",
    "Rill checks her supplies before moving on.",
    "Rill listens for footsteps behind her.",
    "Rill studies the drowned customs house from a distance.",
    "Rill greets the ferryman and asks about the crossing.",
    "Rill rests a hand on the hilt of her knife as she walks.",
)

_AMBIGUOUS_PROMPTS = (
    "Do that again.",
    "Wait, what did you just do?",
    "Go ahead.",
    "Same as before.",
    "Okay, continue.",
)


@pytest.mark.live
@pytest.mark.asyncio
async def test_reasoning_leak_rate_on_ambiguous_and_ordinary_turns(engine):
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
    except (urllib.error.URLError, OSError):
        pytest.skip("the narrator endpoint is not serving")

    from bsh_mcp.service import GameService

    root = engine.config.campaign_root
    game = GameService(root)
    created = game.character_create(
        discord_user_id="live-reasoning-leak-rate-test",
        name="Rill", origin="barbarian",
        backgrounds=["scout", "hunter", "survivor"], weapons=["long knife"],
    )
    assert created["ok"], created
    (root / "campaign" / "players.yaml").write_text(
        "players:\n- discord_user_id: live-reasoning-leak-rate-test\n  character_id: "
        f"{created['character_id']}\n  display_name: Rill\n",
        encoding="utf-8",
    )

    results = []

    def record(category, channel_id, prompt, outcome):
        results.append({
            "category": category,
            "channel_id": channel_id,
            "prompt": prompt,
            "narration": outcome.narration,
            "error": outcome.error,
            "suspect": _suspect(outcome.narration),
        })

    # Category 1: the exact repro -- close an already-closed session.
    for i in range(5):
        channel_id = f"repro-{i}"
        opening_prompt = (
            "Let's end tonight's session here at the ferry landing. "
            f"Title it 'The Ferry Crossing {i}.'"
        )
        opening = await engine.run_turn(InboundTurn(
            channel_id=channel_id, mention=ChannelMessage("Rill", opening_prompt),
        ))
        record("repro-opening", channel_id, opening_prompt, opening)
        closing_prompt = "Yes -- write it up and close the session now."
        closing = await engine.run_turn(InboundTurn(
            channel_id=channel_id, mention=ChannelMessage("Rill", closing_prompt),
        ))
        record("repro-closing", channel_id, closing_prompt, closing)

    # Category 2: other ambiguous, no-antecedent turns.
    for i, prompt in enumerate(_AMBIGUOUS_PROMPTS):
        channel_id = f"ambiguous-{i}"
        outcome = await engine.run_turn(InboundTurn(
            channel_id=channel_id, mention=ChannelMessage("Rill", prompt),
        ))
        record("ambiguous", channel_id, prompt, outcome)

    # Category 3: ordinary declarative turns, no ambiguity.
    for i, prompt in enumerate(_ORDINARY_DECLARATIONS):
        channel_id = f"ordinary-{i}"
        outcome = await engine.run_turn(InboundTurn(
            channel_id=channel_id, mention=ChannelMessage("Rill", prompt),
        ))
        record("ordinary", channel_id, prompt, outcome)

    import json
    Path("/tmp/reasoning-leak-rate-probe.json").write_text(json.dumps(results, indent=2))

    faulted = [r for r in results if r["error"]]
    suspects = [r for r in results if r["suspect"]]
    by_category = {}
    for category in ("repro-opening", "repro-closing", "ambiguous", "ordinary"):
        rows = [r for r in results if r["category"] == category]
        flagged = [r for r in rows if r["suspect"]]
        by_category[category] = f"{len(flagged)}/{len(rows)}"

    summary = (
        f"{len(suspects)}/{len(results)} flagged for review, {len(faulted)} faulted, "
        f"by category: {by_category}"
    )
    print(f"\n{summary}")
    for r in suspects:
        print(f"\n--- SUSPECT [{r['category']}] {r['channel_id']} ---\nprompt: {r['prompt']}\n{r['narration']}")

    assert not faulted, f"{summary}\nfaulted: {faulted}"
