#!/usr/bin/env python3
"""Measure whether the classifier's own token probabilities predict its mistakes.

Nothing here changes production behavior. It sends the production request shape --
the same system prompt, the same prompt builder, the same strict JSON schema, greedy
decode -- through the OpenAI client directly rather than through Strands, because
Strands' ``structured_output`` discards ``logprobs``. The schema is read from the
golden file the contract test pins rather than rebuilt, so this measures the bytes
production sends.

Usage:
    .narrator-venv/bin/python scripts/probe_classifier_logprobs.py         --repeats 5 --report /tmp/classifier-logprobs.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests_narrator"))

from declaration_corpus import ALL, SCENE  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from regression_corpus import REGRESSION_CASES  # noqa: E402

from narrator.classify import CLASSIFIER_SYSTEM_PROMPT, classification_prompt  # noqa: E402
from narrator.config import NarratorConfig  # noqa: E402
from narrator.policy_types import CombatSnapshot, TrustedScope  # noqa: E402

GOLDEN_SCHEMA = REPO_ROOT / "tests_narrator" / "golden" / "classifier_schema.json"

#: The scene ``declaration_corpus`` is labeled against, built exactly as the probe does.
_CORPUS_SCOPE = TrustedScope(
    "session-1", "the-eel-market", tuple(npc for npc, _ in SCENE["present"]),
    ("maren",), tuple(SCENE["present"]),
)

#: The three harvested scopes ``test_probe_regression_corpus`` reconstructs exactly.
_HARVEST_SCOPES: dict[str, tuple[TrustedScope, CombatSnapshot | None]] = {
    "no_one_present": (TrustedScope("session-1", "the-eel-market", ()), None),
    "npc_present": (
        TrustedScope(
            "session-1", "the-eel-market", ("orso-pell",), ("rill",), (("orso-pell", "alive"),)
        ),
        None,
    ),
    "fight": (
        TrustedScope(
            "session-1", "the-eel-market", ("choir-listener",), ("rill",),
            (("choir-listener", "alive"),),
        ),
        CombatSnapshot(active=True, round=1, sides=(("choir-listener", "npc"),)),
    ),
}

#: Harvested labels the classifier deliberately contradicts (see
#: ``test_probe_regression_corpus._DELIBERATE_DISAGREEMENTS``); excluded so a known-better
#: answer is not scored as a miss.
_DELIBERATE = {
    "take the queen from the rusted cabinet",
    "reed thug is punching me",
    "I killed reed thug",
    "I harmed reed thug",
    "Should I open the door, then I stab reed thug?",
    "What if I open the door, then I take his purse?",
}

HAZARDS = ("violence", "theft", "destructive", "none")
ROUTES = ("risk", "social", "read", "out_of_character", "planner")


def _prompts() -> list[dict]:
    """Every (prompt, label) pair the experiment sends, with provenance."""
    items: list[dict] = []
    for case in ALL:
        for language, text in case.text.items():
            items.append({
                "id": f"{case.id}[{language}]",
                "source": "declaration_corpus",
                "text": text,
                "hazard": case.hazard,
                "route": case.route,
                "prompt": classification_prompt(text, _CORPUS_SCOPE, None, False),
            })
    for case_index, case in enumerate(REGRESSION_CASES):
        if case["scope"] not in _HARVEST_SCOPES or case["text"] in _DELIBERATE:
            continue
        scope, combat = _HARVEST_SCOPES[case["scope"]]
        # A harvested case asserts a hazard, or asserts a non-risk route (which is an
        # assertion of no hazard), or asserts neither and is skipped.
        if case["hazard"]:
            hazard = case["hazard"]
        elif case["route"] and case["route"] != "risk":
            hazard = "none"
        else:
            continue
        items.append({
            "id": f"regression:{case_index}",
            "source": "regression_corpus",
            "text": case["text"],
            "hazard": hazard,
            "route": case["route"],
            "prompt": classification_prompt(case["text"], scope, combat, False),
        })
    return items


def _value_margin(content_tokens: list, content: str, field: str, values: tuple[str, ...]) -> dict:
    """The chosen value for ``field`` and its first-token margin over the alternatives.

    Locates the value's first character in the decoded JSON, finds the token covering
    it, and compares that token's logprob against the best ``top_logprobs`` entry that
    is a prefix of a *different* schema-legal value. Tokens the grammar masked report
    at -9999 and never win.
    """
    key = f'"{field}": "'
    start = content.find(key)
    if start < 0:
        return {"value": None, "logprob": None, "alternative": None, "margin": None}
    value_start = start + len(key)
    value_end = content.find('"', value_start)
    chosen = content[value_start:value_end]
    offset = 0
    for token in content_tokens:
        text = token["token"]
        if offset <= value_start < offset + len(text):
            best_alt = None
            for candidate in token.get("top_logprobs", []):
                alt = candidate["token"]
                if alt == text or candidate["logprob"] <= -9000:
                    continue
                for other in values:
                    if other != chosen and other.startswith(alt.strip()) and alt.strip():
                        if best_alt is None or candidate["logprob"] > best_alt:
                            best_alt = candidate["logprob"]
            margin = None if best_alt is None else token["logprob"] - best_alt
            return {
                "value": chosen, "logprob": token["logprob"],
                "alternative": best_alt, "margin": margin,
            }
        offset += len(text)
    return {"value": chosen, "logprob": None, "alternative": None, "margin": None}


async def _one(client: AsyncOpenAI, config: NarratorConfig, schema: dict, item: dict, repeat: int) -> dict:
    started = time.perf_counter()
    response = await client.chat.completions.create(
        model=config.model_id,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ],
        temperature=0.0,
        max_tokens=config.classify_max_tokens,
        logprobs=True,
        top_logprobs=20,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "TurnClassification", "strict": True, "schema": schema},
        },
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    tokens = [t.model_dump() for t in (choice.logprobs.content or [])] if choice.logprobs else []
    try:
        parsed = json.loads(content)
    except ValueError:
        parsed = {}
    return {
        "id": item["id"], "source": item["source"], "repeat": repeat,
        "label_hazard": item["hazard"], "label_route": item["route"],
        "verdict_hazard": parsed.get("hazard"), "verdict_route": parsed.get("route"),
        "hazard": _value_margin(tokens, content, "hazard", HAZARDS),
        "route": _value_margin(tokens, content, "route", ROUTES),
        "defence": _value_margin(tokens, content, "defence_method", DEFENCES),
        "seconds": round(time.perf_counter() - started, 3),
    }


def _summarize(rows: list[dict]) -> dict:
    """Group margins by outcome so the decision reads off one table."""
    by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        by_prompt.setdefault(row["id"], []).append(row)

    def margins(selected: list[dict]) -> dict:
        values = [r["hazard"]["margin"] for r in selected if r["hazard"]["margin"] is not None]
        if not values:
            return {"n": 0}
        values.sort()
        return {
            "n": len(values),
            "min": round(values[0], 2),
            "p10": round(values[len(values) // 10], 2),
            "median": round(statistics.median(values), 2),
            "max": round(values[-1], 2),
        }

    correct, miss, spurious, wrong_kind = [], [], [], []
    for row in rows:
        label, verdict = row["label_hazard"], row["verdict_hazard"]
        if verdict == label:
            correct.append(row)
        elif label != "none" and verdict == "none":
            miss.append(row)
        elif label == "none":
            spurious.append(row)
        else:
            wrong_kind.append(row)
    flipping_ids = {pid for pid, rs in by_prompt.items() if len({r["verdict_hazard"] for r in rs}) > 1}
    stable_correct = [r for r in correct if r["id"] not in flipping_ids]
    flipping = [r for r in rows if r["id"] in flipping_ids]

    # Threshold sweep: at each candidate margin, how many misses would escalate and how
    # many correct answers would escalate with them.
    sweep = []
    for threshold in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        def below(selected):
            return sum(1 for r in selected if r["hazard"]["margin"] is not None and r["hazard"]["margin"] < threshold)
        sweep.append({
            "threshold": threshold,
            "misses_caught": f"{below(miss)}/{len(miss)}",
            "flips_caught": f"{below(flipping)}/{len(flipping)}",
            "correct_escalated": f"{below(correct)}/{len(correct)}",
            "correct_none_escalated": f"{below([r for r in correct if r['label_hazard'] == 'none'])}/{sum(1 for r in correct if r['label_hazard'] == 'none')}",
        })
    return {
        "requests": len(rows),
        "prompts": len(by_prompt),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 3),
        "outcomes": {
            "correct": len(correct), "miss_hazard_as_none": len(miss),
            "spurious_hazard": len(spurious), "wrong_hazard_kind": len(wrong_kind),
        },
        "flipping_prompts": sorted(flipping_ids),
        "margin_by_outcome": {
            "correct_stable": margins(stable_correct),
            "correct_all": margins(correct),
            "miss_hazard_as_none": margins(miss),
            "spurious_hazard": margins(spurious),
            "wrong_hazard_kind": margins(wrong_kind),
            "flipping_prompts": margins(flipping),
        },
        "threshold_sweep": sweep,
        "misses": [
            {"id": r["id"], "repeat": r["repeat"], "margin": r["hazard"]["margin"], "label": r["label_hazard"]}
            for r in miss
        ],
        "spurious": [
            {"id": r["id"], "repeat": r["repeat"], "margin": r["hazard"]["margin"], "verdict": r["verdict_hazard"]}
            for r in spurious
        ],
    }


#: The defence-binding fixture ``test_probe_classifier`` measures, exactly: the
#: player side on the roster and the enemy's turn open, because both were found to
#: change what a bare defence verb reads as. These are the one family the repository
#: documents as stochastic at greedy decode ("je pare le coup" binds about half the
#: time), so they are where a confidence signal would have to earn its place.
_DEFENCE_SCOPE = TrustedScope(
    "session-1", "vey-docks", ("reed-thug",), ("rill",), (("reed-thug", "alive"),)
)
_DEFENCE_FIGHT = CombatSnapshot(
    active=True, round=1, active_actor="reed-thug", sides=(("reed-thug", "npc"), ("rill", "pc"))
)
_DEFENCE_CASES: tuple[tuple[str, str], ...] = (
    ("dodge", "dodge"),
    ("Parry!", "parry"),
    ("I'll try to dodge", "dodge"),
    ("i parry it", "parry"),
    ("j'esquive", "dodge"),
    ("je pare", "parry"),
    ("je pare le coup", "parry"),  # the documented unstable case
    ("ich weiche aus", "dodge"),
    ("уклоняюсь", "dodge"),
    ("парирую", "parry"),
    ("かわす", "dodge"),
    ("受け流す", "parry"),
    ("ich pariere", "parry"),
    ("dodge or parry", "none"),
    ("I hold my ground", "none"),
    ("I attack the reed thug", "none"),
)
DEFENCES = ("dodge", "parry", "none")


def _defence_prompts() -> list[dict]:
    return [
        {
            "id": f"defence:{text}",
            "source": "defence_probe",
            "text": text,
            "hazard": "violence" if text.startswith("I attack") else "none",
            "route": "",
            "defence": expected,
            "prompt": classification_prompt(text, _DEFENCE_SCOPE, _DEFENCE_FIGHT, False),
        }
        for text, expected in _DEFENCE_CASES
    ]


def _summarize_defence(rows: list[dict]) -> dict:
    """Per case: what was bound how often, and the margin behind each answer."""
    by_prompt: dict[str, list[dict]] = {}
    for row in rows:
        by_prompt.setdefault(row["id"], []).append(row)
    cases = []
    for pid, selected in sorted(by_prompt.items()):
        expected = selected[0]["label_defence"]
        counts: dict[str, int] = {}
        margins: dict[str, list[float]] = {}
        for row in selected:
            value = row["defence"]["value"]
            counts[value] = counts.get(value, 0) + 1
            if row["defence"]["margin"] is not None:
                margins.setdefault(value, []).append(round(row["defence"]["margin"], 2))
        cases.append({
            "case": pid, "expected": expected, "counts": counts,
            "hits": f"{counts.get(expected, 0)}/{len(selected)}",
            "margins": {k: sorted(v) for k, v in margins.items()},
        })
    unstable = [c for c in cases if len(c["counts"]) > 1]
    return {
        "requests": len(rows),
        "cases": cases,
        "unstable_cases": [c["case"] for c in unstable],
        "stable_wrong_cases": [c["case"] for c in cases if len(c["counts"]) == 1 and c["hits"].startswith("0/")],
    }


async def _defence_run(args: argparse.Namespace) -> int:
    config = NarratorConfig(campaign_root=REPO_ROOT)
    schema = json.loads(GOLDEN_SCHEMA.read_text(encoding="utf-8"))
    client = AsyncOpenAI(base_url=config.base_url, api_key="not-required")
    semaphore = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    async def bounded(item: dict, repeat: int) -> None:
        async with semaphore:
            row = await _one(client, config, schema, item, repeat)
            row["label_defence"] = item["defence"]
            rows.append(row)

    await asyncio.gather(*(bounded(item, r) for item in _defence_prompts() for r in range(args.repeats)))
    rows.sort(key=lambda r: (r["id"], r["repeat"]))
    summary = _summarize_defence(rows)
    summary["repeats"] = args.repeats
    Path(args.report).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


async def _main(args: argparse.Namespace) -> int:
    if args.defence:
        return await _defence_run(args)
    config = NarratorConfig(campaign_root=REPO_ROOT)
    schema = json.loads(GOLDEN_SCHEMA.read_text(encoding="utf-8"))
    client = AsyncOpenAI(base_url=config.base_url, api_key="not-required")
    items = _prompts()
    if args.limit:
        items = items[: args.limit]
    semaphore = asyncio.Semaphore(args.concurrency)
    rows: list[dict] = []

    async def bounded(item: dict, repeat: int) -> None:
        async with semaphore:
            try:
                rows.append(await _one(client, config, schema, item, repeat))
            except Exception as error:  # noqa: BLE001 - a fault is a data point here
                rows.append({
                    "id": item["id"], "source": item["source"], "repeat": repeat,
                    "label_hazard": item["hazard"], "label_route": item["route"],
                    "verdict_hazard": None, "verdict_route": None,
                    "hazard": {"value": None, "margin": None}, "route": {"value": None, "margin": None},
                    "seconds": 0.0, "fault": repr(error)[:200],
                })

    started = time.perf_counter()
    await asyncio.gather(*(bounded(item, repeat) for item in items for repeat in range(args.repeats)))
    elapsed = round(time.perf_counter() - started, 1)
    rows.sort(key=lambda r: (r["id"], r["repeat"]))
    summary = _summarize(rows)
    summary["wall_seconds"] = elapsed
    summary["repeats"] = args.repeats
    report = {"summary": summary, "rows": rows}
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="send only the first N prompts")
    parser.add_argument(
        "--defence", action="store_true",
        help="run the defence-binding fixture instead of the corpus, measuring defence_method",
    )
    parser.add_argument("--report", default="/tmp/classifier-logprobs.json")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
