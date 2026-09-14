"""Live acceptance for the hazard assessor, against the served model endpoint.

The assessor's one job is telling a declaration from a report about one, and that is
a model behaviour, which fake-engine tests cannot certify. This probe measures the
exact production configuration -- ``NarratorEngine.assess_hazard``'s greedy decode,
schema, system prompt, and prompt builder -- against the served endpoint, over the
recorded live-session shapes the lexical floor misjudged plus the declarations it
must keep.

The offline staleness guard below asserts against ``lexical_double.classify_turn``,
the retired English classifier kept as test code, rather than against production
routing, which is now the model classifier in ``narrator.classify``. That is the right
thing to assert there: this corpus was harvested from what the *lexical* trigger sent
the assessor, and the guard's whole job is documenting which messages the assessor
historically received. Asserting it against the model would make an offline guard
depend on a live endpoint, and would silently rewrite the corpus's provenance the
first time the two classifiers disagreed. What production routes to the assessor now
is measured by ``tests_narrator/test_probe_classifier.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lexical_double import classify_turn  # noqa: E402

from narrator.config import NarratorConfig  # noqa: E402
from narrator.engine import NarratorEngine  # noqa: E402
from narrator.policy_types import TrustedScope  # noqa: E402

_SCOPE = TrustedScope(
    "session-1", "the-eel-market", ("rade", "orso-pell"), ("maren",),
    (("rade", "alive"), ("orso-pell", "alive")),
)

#: First-strike declarations: the ask is owed, so the assessor must answer
#: ``declaration`` for every one.
MUST_KEEP = (
    "attack rade",
    "I stab the merchant",
    "Punch a trader",
    "I punch the trader in the face and take the rope",
    "kill the smuggler",
    "I burn the ledger",
    "I steal his purse",
    "I attack him",
    "stab the guard and take the lamp",
    "I strangle the clerk with my rope",
)


MUST_DOWNGRADE = (
    "i attacked him",
    "gm i had attacked and killed him. look at the transcript",
    "the guard attacked me",
    "I attacked the fishmonger already, check the log",
    "she killed my brother years ago",
    "the thugs killed the bell-keeper before we arrived",
    "he attacked me first, I swear",
)


def test_the_corpus_is_what_the_assessor_actually_receives():
    """Offline staleness guard: every corpus message must route ``risk`` through the
    lexical double, because this corpus is exactly the set of messages the lexical
    trigger flagged and handed the assessor. A case that stops routing risk would
    silently stop measuring anything. See the module docstring for why the double,
    and not production's model classifier, is the right thing to assert against."""
    for text in MUST_KEEP + MUST_DOWNGRADE:
        assert classify_turn(text, scope=_SCOPE).route == "risk", text


@pytest.mark.live
async def test_the_live_assessor_keeps_every_declaration_and_drops_every_report(tmp_path: Path):
    engine = NarratorEngine(NarratorConfig(campaign_root=tmp_path))
    engine._client = object()  # noqa: SLF001 - started-engine seam; only the model endpoint is used

    misses = []
    for text in MUST_KEEP:
        assessment = await engine.assess_hazard(text, scope=_SCOPE)
        if assessment is None or assessment.kind != "declaration":
            misses.append((text, assessment.kind if assessment else "FAULT"))
    assert not misses, f"a declaration was downgraded, which skips a table-safety ask: {misses}"

    misses = []
    for text in MUST_DOWNGRADE:
        assessment = await engine.assess_hazard(text, scope=_SCOPE)
        if assessment is None or assessment.kind == "declaration":
            misses.append((text, assessment.kind if assessment else "FAULT"))
    assert not misses, f"a report kept its confirmation (spurious ask, not a safety hole): {misses}"
