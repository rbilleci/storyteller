"""Test probe scene entities.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.config import NarratorConfig  # noqa: E402
from narrator.engine import NarratorEngine, _character_resource_values  # noqa: E402

pytestmark = pytest.mark.live

REPEATS = int(os.environ.get("BSH_ENTITY_PROBE_REPEATS", "10"))
EVIDENCE_DIR = Path(os.environ.get("BSH_ENTITY_PROBE_EVIDENCE", "/tmp/scene-entities-probe"))


FRAME_NARRATION = (
    'A narrow balcony holds an idle brass pulley. Two iron brackets attach it to a granite post.'
)


ATMOSPHERE_NARRATION = (
    "The wind picks up off the mud flats. Gulls wheel over the silent tower, and the "
    "light goes the color of old pewter. Nobody speaks for a while; the planks creak "
    "under the weight of the afternoon."
)

#: The clerk the live scene record carried five facts about while Present NPCs read
#: "None recorded" -- Slice B's introduction case.
CLERK_NARRATION = (
    "A Salt Magistrate clerk in a salt-stained coat picks his way along the planks "
    "toward Rill, a tally slate under one arm. He stops two paces off and looks Rill "
    "up and down without a word, then makes a small mark on the slate."
)

#: Second iteration. The Slice B soak introduced "Sera Vane, the bell-keeper" under
#: its own slug beside the authored ``sera-vane``: the label must still be captured
#: (the server converges it; pinned offline), and the roster arm must not drop it.
EPITHET_NARRATION = (
    "Along the planks from the road-shrine side comes Sera Vane, the bell-keeper, a "
    "shuttered lamp in one hand and two days of mud on her hem. She looks at Rill for "
    "a long moment, then up at the silent bell."
)


DOOR_NARRATION = (
    "The tower door scrapes open and Sera Vane, the bell-keeper, steps out onto the "
    "planks with a shuttered lamp in one hand. She looks at Rill for a long moment, "
    "then at the silent bell above."
)

#: Second iteration. The same soak recorded "low-level officials of the Salt
#: Magistrate" as a person. A crowd and a plural category must introduce nobody.
CROWD_NARRATION = (
    "A knot of Salt Magistrate officials moves along the planks, counting barrels and "
    "saying little. Behind them a crowd of fishwives has gathered to watch, muttering "
    "about the bell. Nobody steps forward."
)

SCENE_SUMMARY = "The party stands in the eel market at low water, under the silent bell tower."

#: Open question 3, the persons half: Vell is a recorded person in the probe campaign;
#: a narration in which she leaves must reach ``scene_commit`` as ``departed_persons``.
DEPARTURE_NARRATION = (
    "Vell the eel-seller shutters her stall, tucks the day's coins into her apron, and "
    "walks off along the planks toward the customs house without a word to anyone."
)

#: spec-language-independent-guards.md §3.5, the numeric half, measured live. Rill
#: holds 5 coins and 9 hit points in the probe campaign. A fabricated total in French
#: or Japanese reads nothing through the English regexes; the typed fields must catch
#: it. A true figure must still ratify, and a digit about the world must not trip.
NUMERIC_CASES = {
    "coins-fr-false": ("Après l'échange, Rill a maintenant 40 pièces de cuivre dans sa bourse.", "none"),
    "coins-fr-true": ("Après l'échange, Rill a maintenant 5 pièces de cuivre dans sa bourse.", "record"),
    "hp-ja-false": ("リルは傷を負った。リルの体力は残り3です。", "none"),
    "tide-digit": (
        'Rill studies a harbor sign: high water is expected 2 hours after sunset.',
        "record",
    ),
}


RESURRECTIONS = {
    "en": "Rade is alive beside a red cart.",
    "fr": "Rade est vivant près d’une charrette rouge.",
    "de": "Rade ist am Leben neben einem roten Wagen.",
    "ja": "ラデは生きていて、赤い荷車のそばにいる。",
    "ru": "Раде жив рядом с красной телегой.",
}


def _campaign(root: Path) -> None:
    campaign = root / "campaign"
    (campaign / "characters").mkdir(parents=True, exist_ok=True)
    (campaign / "state.json").write_text(
        json.dumps(
            {
                "fiction_debt": [],
                "npcs": {
                    "rade": {"id": "rade", "name": "Rade", "level": 1, "hp": 0, "hp_max": 5,
                             "damage": 4, "status": "dead", "location_id": "the-eel-market"},
                    "orso-pell": {"id": "orso-pell", "name": "Orso Pell", "level": 3, "hp": 12,
                                  "hp_max": 12, "damage": 6, "status": "alive"},
                },
                "scene": {
                    "location_id": "the-eel-market",
                    "summary": SCENE_SUMMARY,
                    "objects": {"tower-door": {"id": "tower-door", "state": "barred"}},
                    "exits": ["the-drowned-customs-house", "the-road-shrine"],
                    "persons": {
                        "vell-the-eel-seller": {"id": "vell-the-eel-seller", "name": "Vell the eel-seller",
                                                "source": "model", "first_event": 3, "last_event": 3},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    for character_id, name in (("ossa", "Ossa"), ("rill", "Rill")):
        (campaign / "characters" / f"{character_id}.json").write_text(
            json.dumps({"id": character_id, "name": name, "coins": 5, "hp": 9, "equipment": ["rope"]}),
            encoding="utf-8",
        )


class _Probe:
    """One engine, its sweep spied on and its tool call captured."""

    def __init__(self, root: Path) -> None:
        self.engine = NarratorEngine(NarratorConfig(campaign_root=root))
        self.engine._client = object()  # noqa: SLF001 - started-engine seam; only the model endpoint is used
        self.engine._resources_before_turn = _character_resource_values(root)  # noqa: SLF001
        self.raw: list = []
        self.calls: list[dict] = []
        original = self.engine._sweep_once  # noqa: SLF001

        async def spy(prompt):
            outcome = await original(prompt)
            self.raw.append(outcome)
            return outcome

        self.engine._sweep_once = spy  # noqa: SLF001
        self.engine._call_tool = (  # noqa: SLF001
            lambda name, arguments, origin="settle": self.calls.append(dict(arguments)) or "success"
        )

    async def run(self, case: str, narration: str, repeats: int) -> list[dict]:
        rows = []
        for repeat in range(repeats):
            self.raw.clear()
            self.calls.clear()
            started = time.perf_counter()
            kind = await self.engine._sweep(narration)  # noqa: SLF001
            seconds = time.perf_counter() - started
            raw = self.raw[0].model_dump() if self.raw else None
            rows.append(
                {
                    "case": case, "repeat": repeat, "kind": kind,
                    "seconds": round(seconds, 3),
                    "model_kind": (raw or {}).get("outcome", {}).get("kind"),
                    "public_summary": (raw or {}).get("outcome", {}).get("public_summary"),
                    "visible_changes": (raw or {}).get("outcome", {}).get("visible_changes"),
                    "mentions": (raw or {}).get("outcome", {}).get("mentions"),
                    "introduced_persons": (raw or {}).get("outcome", {}).get("introduced_persons"),
                    "persons_committed": (self.calls[0].get("persons") if self.calls else None),
                    "departed_committed": (self.calls[0].get("departed_persons") if self.calls else None),
                    "summary_committed": (self.calls[0].get("public_summary") if self.calls else None),
                    "claimed_coins": (raw or {}).get("outcome", {}).get("claimed_coins"),
                    "claimed_hp": (raw or {}).get("outcome", {}).get("claimed_hp"),
                }
            )
        return rows


async def _measure(root: Path) -> dict:
    _campaign(root)
    rows: list[dict] = []
    probe = _Probe(root)
    rows += await probe.run("frame", FRAME_NARRATION, REPEATS)
    rows += await probe.run("atmosphere", ATMOSPHERE_NARRATION, REPEATS)
    for language, text in RESURRECTIONS.items():
        rows += await probe.run(f"resurrection-{language}", text, REPEATS)
    rows += await probe.run("clerk", CLERK_NARRATION, REPEATS)
    rows += await probe.run("epithet", EPITHET_NARRATION, REPEATS)
    rows += await probe.run("crowd", CROWD_NARRATION, REPEATS)
    rows += await probe.run("door", DOOR_NARRATION, REPEATS)
    for case_id, (narration, _expected) in NUMERIC_CASES.items():
        rows += await probe.run(case_id, narration, REPEATS)
    rows += await probe.run("departure", DEPARTURE_NARRATION, REPEATS)

    def count(case: str, predicate) -> int:
        return sum(1 for row in rows if row["case"] == case and predicate(row))

    seconds = [row["seconds"] for row in rows]
    summary: dict = {
        # The sweep's own verdict (what case 1 gates) and the final verdict after
        # the guard, reported side by side: the first run of this probe found the
        # two diverging on the frame narration because ``states_new_arrival``
        # fires on "the party reaches" and not on "the players reach".
        "frame_model_records": count("frame", lambda r: r["model_kind"] == "record"),
        "frame_records": count("frame", lambda r: r["kind"] == "record"),
        "atmosphere_model_declines": count("atmosphere", lambda r: r["model_kind"] == "none"),
        "atmosphere_declines": count("atmosphere", lambda r: r["kind"] == "none"),
        "resurrection_declined": {
            language: count(f"resurrection-{language}", lambda r: r["kind"] == "none")
            for language in RESURRECTIONS
        },
        "resurrection_model_records": {
            language: count(f"resurrection-{language}", lambda r: r["model_kind"] == "record")
            for language in RESURRECTIONS
        },
        "resurrection_alive_mentions": {
            language: count(
                f"resurrection-{language}",
                lambda r: any(
                    m.get("entity_id") == "rade" and m.get("claim") == "alive"
                    for m in (r["mentions"] or [])
                ),
            )
            for language in RESURRECTIONS
        },
        "latency_median_seconds": round(statistics.median(seconds), 3),
        "latency_max_seconds": round(max(seconds), 3),
        "repeats": REPEATS,
    }
    summary["clerk_introduced"] = count(
        "clerk",
        lambda r: any("clerk" in str(p.get("name", "")).lower() for p in (r["persons_committed"] or [])),
    )
    summary["clerk_records"] = count("clerk", lambda r: r["kind"] == "record")
    summary["clerk_model_records"] = count("clerk", lambda r: r["model_kind"] == "record")
    # Second iteration, change 3: a frame record whose facts were kept under the
    # scene's own summary after the arrival guard fired on its headline.
    summary["frame_salvaged"] = count(
        "frame", lambda r: r["kind"] == "record" and r["summary_committed"] == SCENE_SUMMARY
    )
    summary["clerk_named_in_extraction"] = count(
        "clerk", lambda r: any("clerk" in str(p).lower() for p in (r["introduced_persons"] or []))
    )
    summary["atmosphere_introductions"] = count(
        "atmosphere", lambda r: bool(r["introduced_persons"]) or bool(r["persons_committed"])
    )
    summary["epithet_captured"] = count(
        "epithet",
        lambda r: any("sera vane" in str(p.get("name", "")).lower() for p in (r["persons_committed"] or [])),
    )
    summary["crowd_introductions"] = count(
        "crowd", lambda r: bool(r["introduced_persons"]) or bool(r["persons_committed"])
    )
    summary["crowd_labels"] = sorted(
        {str(p) for r in rows if r["case"] == "crowd" for p in (r["introduced_persons"] or [])}
    )
    summary["numeric"] = {
        case_id: {
            "expected": expected,
            "final_as_expected": count(case_id, lambda r, e=expected: r["kind"] == e),
            "model_records": count(case_id, lambda r: r["model_kind"] == "record"),
            "claimed_coins": dict(Counter(str(r["claimed_coins"]) for r in rows if r["case"] == case_id)),
            "claimed_hp": dict(Counter(str(r["claimed_hp"]) for r in rows if r["case"] == case_id)),
        }
        for case_id, (_n, expected) in NUMERIC_CASES.items()
    }
    summary["departure_committed"] = count(
        "departure", lambda r: "vell-the-eel-seller" in (r["departed_committed"] or [])
    )
    summary["departure_records"] = count("departure", lambda r: r["kind"] == "record")
    summary["door_declined"] = count("door", lambda r: r["kind"] == "none")
    summary["door_open_mentions"] = count(
        "door",
        lambda r: any(m.get("entity_id") == "tower-door" and m.get("claim") == "open" for m in (r["mentions"] or [])),
    )
    return {"repeats": REPEATS, "rows": rows, "summary": summary}


@pytest.fixture(scope="module")
def measured(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("scene-entities")
    results = asyncio.run(_measure(root))
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / "scene-entities-probe.json"
    text = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    results["evidence_path"] = str(path)
    results["evidence_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"\nevidence: {path} sha256 {results['evidence_sha256']}")
    print(json.dumps(results["summary"], indent=2, ensure_ascii=False))
    return results


def test_case_1_the_sweep_verdict_holds_under_the_roster(measured):
    """Record/decline stability of the sweep's own verdict: the frame records and the
    atmosphere declines at the recorded ceiling. Gated on the model's verdict, not the
    guard's final one: the guard's regexes read the record *text*, and the first run
    of this probe measured the roster changing that text's conjugation ("the party
    reaches" against "the players reach") without changing the verdict -- see the
    evidence file's ``frame_records`` beside ``frame_model_records``."""
    summary = measured["summary"]
    assert summary["frame_model_records"] >= REPEATS - 1, summary
    assert summary["atmosphere_model_declines"] == REPEATS, summary


def test_case_2_the_resurrection_is_declined_in_every_locale(measured):
    """The D4 laundering, closed by identifier: with the roster, every language's
    final verdict is ``none`` at the ceiling."""
    declined = measured["summary"]["resurrection_declined"]
    assert all(value == REPEATS for value in declined.values()), declined


def test_case_3_the_clerk_is_introduced(measured):
    summary = measured["summary"]
    assert summary["clerk_introduced"] >= REPEATS - 1, summary


def test_case_4_atmosphere_introduces_nobody(measured):
    summary = measured["summary"]
    assert summary["atmosphere_introductions"] == 0, summary


def test_case_5_latency_is_recorded(measured):
    summary = measured["summary"]
    assert summary["latency_median_seconds"] > 0, summary
    assert summary["latency_max_seconds"] < NarratorConfig.sweep_timeout_seconds, summary


def test_case_7_the_frame_detail_survives_the_arrival_headline(measured):
    """Second iteration, change 3: the model heads the frame record with the party's
    arrival; the guard keeps the timber-frame fact under the scene's current summary
    instead of declining the record whole."""
    summary = measured["summary"]
    # The detail must reach the record; whether it needed salvaging depends on
    # whether the model headed the record with an arrival verb that turn
    # ("reaches" does, "approaches" does not), so ``frame_salvaged`` is reported
    # beside the gate rather than gated.
    assert summary["frame_records"] >= REPEATS - 1, summary


def test_case_8_an_epithet_label_is_still_captured(measured):
    summary = measured["summary"]
    assert summary["epithet_captured"] >= REPEATS - 1, summary


def test_case_9_a_crowd_introduces_nobody(measured):
    summary = measured["summary"]
    assert summary["crowd_introductions"] == 0, summary


def test_case_10_a_barred_door_narrated_open_does_not_reach_canon(measured):
    summary = measured["summary"]
    assert summary["door_declined"] == REPEATS, summary
    assert summary["door_open_mentions"] >= REPEATS - 1, summary


def test_case_11_typed_figures_verify_in_any_language_and_a_world_digit_does_not_trip(measured):
    numeric = measured["summary"]["numeric"]
    bad = {case: figures for case, figures in numeric.items() if figures["final_as_expected"] < REPEATS - 1}
    assert not bad, bad


def test_case_12_a_person_who_walks_off_leaves_the_record(measured):
    summary = measured["summary"]
    assert summary["departure_committed"] >= REPEATS - 1, summary
