#!/usr/bin/env python3
"""Measure the two open classifier residuals and the narrow compositions that close them.

``--mode variants``
    The in-prompt fix for the trailing-question theft, placed in the hazard block (V1)
    and on the ``accepts_offer`` line (V2), against the baseline, with the defence
    bindings in the same run because placement effects are diffuse. Ten repeats.

``--mode escalation``
    The typed trigger and the narrow compositions. Every prompt -- baseline and narrow,
    hazard and defence -- is shuffled into one mixed batch, because the defence
    bindings were found to depend on what else is in the batch. Prints the
    ``accepts_offer`` / ``named_person_ids`` signature beside each hazard verdict so the
    escalation trigger's presence is visible, including on the fiction-offered gifts
    that must not escalate into a confirmation. Ten repeats.

``--mode engine``
    The production path end to end: ``NarratorEngine.classify_intent`` with its
    companions, every hazard and defence case shuffled into one mixed batch, so the
    number measured is the one a table gets. Ten repeats.

``--mode parry``
    The French family at twenty repeats, sequential then concurrent, then through the
    defence-only composition. This is the measurement that separates a wording residual
    from a batching one.

Usage:
    .narrator-venv/bin/python scripts/probe_classifier_escalation.py --mode escalation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from openai import AsyncOpenAI  # noqa: E402
from openai.lib._pydantic import to_strict_json_schema  # noqa: E402

from narrator.classify import (  # noqa: E402
    CLASSIFIER_SYSTEM_PROMPT,
    DEFAULT_TURN_CLASSIFIER,
    FACETS,
    PROVIDERS,
    RESOLVERS,
    DefenceFacet,
    HazardFacet,
    OfferFacet,
    ReferentFacet,
    RouteFacet,
    build_turn_classifier,
)
from narrator.config import NarratorConfig  # noqa: E402
from narrator.facets import ClassificationContext  # noqa: E402
from narrator.policy_types import CombatSnapshot, TrustedScope  # noqa: E402

MARKET = TrustedScope(
    "session-1", "the-eel-market", ("rade", "orso-pell", "maren"), ("maren",),
    (("rade", "alive"), ("orso-pell", "alive"), ("maren", "alive")),
)
NPC_PRESENT = TrustedScope(
    "session-1", "the-eel-market", ("orso-pell",), ("rill",), (("orso-pell", "alive"),)
)
NO_ONE = TrustedScope("session-1", "the-eel-market", ())
DEF_SCOPE = TrustedScope(
    "session-1", "vey-docks", ("reed-thug",), ("rill",), (("reed-thug", "alive"),)
)
DEF_FIGHT = CombatSnapshot(
    active=True, round=1, active_actor="reed-thug", sides=(("reed-thug", "npc"), ("rill", "pc"))
)

#: The candidate rule: acceptance needs the open-offer statement. Measured to fix the
#: theft targets and to cost the German parry when placed in the combined prompt.
OFFER_RULE = (
    "No offer has been made to the character unless the scene above says one is open. "
    "Without that statement, taking something from a person is taking it, not accepting "
    "it: 'I take it from him' is theft, whatever question follows."
)


@dataclass(frozen=True)
class HazardFacetWithOfferRule(HazardFacet):
    """The hazard block with the rule inserted after the trailing-question paragraph."""

    name: str = "hazard"

    def rules(self) -> list[str]:
        base = HazardFacet.rules(self)
        idx = next(i for i, line in enumerate(base) if line.startswith("A declaration keeps its hazard"))
        return base[: idx + 2] + [OFFER_RULE, ""] + base[idx + 2:]


@dataclass(frozen=True)
class OfferFacetWithRule(OfferFacet):
    """The rule appended to the ``accepts_offer`` definition line instead."""

    name: str = "offer"

    def rules(self) -> list[str]:
        base = OfferFacet.rules(self)
        return [base[0] + " When the scene above does not say an offer is open, nothing has "
                "been offered: a message that takes something from a person is taking, never "
                "accepting, whatever question follows it."] + base[1:]


def _swap(facets, **by_name):
    return tuple(by_name.get(facet.name, facet) for facet in facets)


def _compose(name, facets, resolvers=()):
    return build_turn_classifier(
        system_prompt=CLASSIFIER_SYSTEM_PROMPT, providers=PROVIDERS, facets=facets,
        resolvers=resolvers, schema_name=name,
    )


VARIANTS = {
    "baseline": DEFAULT_TURN_CLASSIFIER,
    "v1-hazard-block": _compose("V1", _swap(FACETS, hazard=HazardFacetWithOfferRule()), RESOLVERS),
    "v2-offer-line": _compose("V2", _swap(FACETS, offer=OfferFacetWithRule()), RESOLVERS),
}
HAZARD_NARROW = _compose(
    "HazardOnly", (RouteFacet(), HazardFacetWithOfferRule(), ReferentFacet(), OfferFacet())
)
DEFENCE_NARROW = _compose("DefenceOnly", (RouteFacet(), DefenceFacet()))

# (text, scope, combat, offer_open, field, expected)
HAZARD_CASES = [
    ("I take it from rade. What next?", MARKET, None, False, "hazard", "theft"),
    ("I take it from orso-pell. What next?", NPC_PRESENT, None, False, "hazard", "theft"),
    ("I take it from him. What next?", NPC_PRESENT, None, False, "hazard", "theft"),
    ("Ich nehme es von rade. Was nun?", MARKET, None, False, "hazard", "theft"),
    ("I take the lantern from rade. What next?", MARKET, None, False, "hazard", "theft"),
    ("I grab the lantern from rade's stall", MARKET, None, False, "hazard", "theft"),
    # Fiction-offered gifts carry the same typed signature and must stay ``none``.
    ("I take the coin rade offers me", MARKET, None, False, "hazard", "none"),
    ("I accept the drink from rade", MARKET, None, False, "hazard", "none"),
    ("I take the lantern rade holds out to me. What next?", MARKET, None, False, "hazard", "none"),
    ("Rade hands me the key and I take it", MARKET, None, False, "hazard", "none"),
    # Controls.
    ("I take it from rade. What next?", MARKET, None, True, "hazard", "none"),
    ("I take it", MARKET, None, True, "accepts_offer", True),
    ("Je le prends", MARKET, None, True, "accepts_offer", True),
    ("I take the rope from the rack", MARKET, None, False, "hazard", "none"),
    ("I take a break by the stall", MARKET, None, False, "hazard", "none"),
    ("I take it. What next?", NO_ONE, None, False, "hazard", "none"),
]
DEFENCE_CASES = [
    (text, DEF_SCOPE, DEF_FIGHT, False, "defence_method", expected)
    for text, expected in (
        ("dodge", "dodge"), ("Parry!", "parry"), ("I'll try to dodge", "dodge"),
        ("i parry it", "parry"), ("j'esquive", "dodge"), ("je pare", "parry"),
        ("je pare le coup", "parry"), ("ich weiche aus", "dodge"), ("ich pariere", "parry"),
        ("уклоняюсь", "dodge"), ("парирую", "parry"), ("かわす", "dodge"), ("受け流す", "parry"),
        ("dodge or parry", "none"), ("I hold my ground", "none"),
        ("I attack the reed thug", "none"),
        ("I dodge his swing and stab the barkeep", "none"),
        ("parry then kill the priest", "none"),
    )
]
PARRY_FAMILY = [
    (text, DEF_SCOPE, DEF_FIGHT, False, "defence_method", expected)
    for text, expected in (
        ("je pare le coup", "parry"), ("je pare son coup", "parry"), ("je pare", "parry"),
        ("ich pariere den Schlag", "parry"), ("ich pariere", "parry"), ("I parry the blow", "parry"),
        ("dodge or parry", "none"), ("I attack the reed thug", "none"), ("I hold my ground", "none"),
    )
]


async def _ask(client, config, classifier, schema, text, scope, combat, offer):
    prompt = classifier.prompt(text, ClassificationContext(scope=scope, combat=combat, offer_open=offer))
    response = await client.chat.completions.create(
        model=config.model_id, temperature=0.0, max_tokens=config.classify_max_tokens,
        messages=[
            {"role": "system", "content": classifier.system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": classifier.schema.__name__, "strict": True, "schema": schema},
        },
    )
    return json.loads(response.choices[0].message.content or "{}")


def _report(label, results):
    print(f"=== {label} ===")
    for (text, offer, field), rows in results.items():
        expected = rows[0][1]
        hits = sum(1 for got, _, _ in rows if got == expected)
        others: dict = {}
        for got, _, _ in rows:
            if got != expected:
                others[got] = others.get(got, 0) + 1
        signature = ""
        if field == "hazard":
            accepts = sum(1 for _, _, v in rows if v.get("accepts_offer"))
            named = sum(1 for _, _, v in rows if v.get("named_person_ids"))
            signature = f"  accepts_offer={accepts}/{len(rows)} named={named}/{len(rows)}"
        flag = "" if hits == len(rows) else "   <--"
        print(
            f"  {hits:>2}/{len(rows)} want={expected!s:<6} [offer={offer}] {text}"
            f"{'  other=' + str(others) if others else ''}{signature}{flag}"
        )


async def _batch(client, config, jobs, concurrency, shuffle=False):
    """Run (label, classifier, case) jobs; returns {label: {(text, offer, field): rows}}."""
    if shuffle:
        random.Random(7).shuffle(jobs)
    schemas = {id(clf): to_strict_json_schema(clf.schema) for _, clf, _ in jobs}
    semaphore = asyncio.Semaphore(concurrency)
    out: dict = {}

    async def one(label, classifier, case):
        text, scope, combat, offer, field, expected = case
        async with semaphore:
            verdict = await _ask(client, config, classifier, schemas[id(classifier)], text, scope, combat, offer)
        out.setdefault(label, {}).setdefault((text, offer, field), []).append((verdict.get(field), expected, verdict))

    started = time.perf_counter()
    await asyncio.gather(*(one(*job) for job in jobs))
    print(f"{len(jobs)} requests, {time.perf_counter() - started:.0f}s, concurrency {concurrency}\n")
    return out


async def _main(args) -> int:
    config = NarratorConfig(campaign_root=REPO_ROOT)
    client = AsyncOpenAI(base_url=config.base_url, api_key="not-required")
    n = args.repeats
    if args.mode == "variants":
        cases = HAZARD_CASES + DEFENCE_CASES[4:9]
        for label, classifier in VARIANTS.items():
            out = await _batch(client, config, [(label, classifier, c) for c in cases for _ in range(n)], 8)
            _report(label, out[label])
    elif args.mode == "escalation":
        jobs = []
        for label, classifier, cases in (
            ("baseline", DEFAULT_TURN_CLASSIFIER, HAZARD_CASES + DEFENCE_CASES),
            ("hazard-narrow", HAZARD_NARROW, HAZARD_CASES),
            ("defence-narrow", DEFENCE_NARROW, DEFENCE_CASES),
        ):
            jobs += [(label, classifier, c) for c in cases for _ in range(n)]
        out = await _batch(client, config, jobs, 8, shuffle=True)
        for label in ("baseline", "hazard-narrow", "defence-narrow"):
            _report(label, out[label])
    elif args.mode == "engine":
        from narrator.engine import NarratorEngine

        engine = NarratorEngine(config)
        engine._client = object()  # noqa: SLF001 - started-engine seam; only the endpoint is used
        semaphore = asyncio.Semaphore(8)
        out: dict = {}
        cases = HAZARD_CASES + DEFENCE_CASES
        jobs = [case for case in cases for _ in range(n)]
        random.Random(7).shuffle(jobs)

        async def one(case):
            text, scope, combat, offer, field, expected = case
            async with semaphore:
                verdict = await engine.classify_intent(text, scope=scope, combat=combat, offer_open=offer)
            value = None if verdict is None else getattr(verdict, field)
            payload = {} if verdict is None else verdict.model_dump()
            out.setdefault((text, offer, field), []).append((value, expected, payload))

        started = time.perf_counter()
        await asyncio.gather(*(one(case) for case in jobs))
        print(f"{len(jobs)} turns through classify_intent, {time.perf_counter() - started:.0f}s\n")
        _report("engine: classify_intent with companions", out)
        tally: dict = {}
        for record in engine._companion_log:  # noqa: SLF001
            key = (record.get("defence"), record.get("hazard"))
            tally[key] = tally.get(key, 0) + 1
        print("companion log (defence, hazard):", dict(sorted(tally.items(), key=str)))
    elif args.mode == "parry":
        for label, classifier, concurrency in (
            ("baseline sequential", DEFAULT_TURN_CLASSIFIER, 1),
            ("baseline concurrent", DEFAULT_TURN_CLASSIFIER, 8),
            ("defence-narrow concurrent", DEFENCE_NARROW, 8),
        ):
            out = await _batch(client, config, [(label, classifier, c) for c in PARRY_FAMILY for _ in range(2 * n)], concurrency)
            _report(label, out[label])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("variants", "escalation", "engine", "parry"), default="escalation")
    parser.add_argument("--repeats", type=int, default=10)
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
