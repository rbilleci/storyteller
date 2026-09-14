#!/usr/bin/env python3
"""Regenerate the classifier and assessor goldens under ``tests_narrator/golden/``.

Run this only after the live probes have accepted a prompt change at ten repeats per
case, and record that measurement in ``tests_narrator/golden/ACCEPTANCE.md`` in the
same change. ``tests_narrator/test_classifier_contract.py`` reads what this writes, and
its docstring says why a golden diff is a prompt change rather than a test to update.

The scene configurations are declared in the contract test and imported from it, so the
two cannot drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests_narrator"))

from openai.lib._pydantic import to_strict_json_schema  # noqa: E402
from test_classifier_contract import (  # noqa: E402
    COMPANION_CASES,
    DECLARATION,
    GOLDEN,
    GOLDEN_CASES,
)

from narrator.assess import (  # noqa: E402
    ASSESSOR_SYSTEM_PROMPT,
    HazardAssessment,
    assessment_prompt,
)
from narrator.classify import (  # noqa: E402
    CLASSIFIER_SYSTEM_PROMPT,
    DEFENCE_COMPANION,
    HAZARD_COMPANION,
    PERSONS_COMPANION,
    ROUTE_COMPANION,
    TurnClassification,
    classification_prompt,
)
from narrator.facets import ClassificationContext  # noqa: E402


def main() -> int:
    GOLDEN.mkdir(exist_ok=True)
    for name, (scope, combat, offer) in GOLDEN_CASES.items():
        (GOLDEN / f"classifier_prompt.{name}.txt").write_text(
            classification_prompt(DECLARATION, scope, combat, offer), encoding="utf-8"
        )
        (GOLDEN / f"assessor_prompt.{name}.txt").write_text(
            assessment_prompt(DECLARATION, scope, combat), encoding="utf-8"
        )
    for name, (scope, combat, offer) in COMPANION_CASES.items():
        context = ClassificationContext(scope=scope, combat=combat, offer_open=offer)
        (GOLDEN / f"defence_companion_prompt.{name}.txt").write_text(
            DEFENCE_COMPANION.prompt(DECLARATION, context), encoding="utf-8"
        )
        (GOLDEN / f"hazard_companion_prompt.{name}.txt").write_text(
            HAZARD_COMPANION.prompt(DECLARATION, context), encoding="utf-8"
        )
        (GOLDEN / f"persons_companion_prompt.{name}.txt").write_text(
            PERSONS_COMPANION.prompt(DECLARATION, context), encoding="utf-8"
        )
        (GOLDEN / f"route_companion_prompt.{name}.txt").write_text(
            ROUTE_COMPANION.prompt(DECLARATION, context), encoding="utf-8"
        )
    for filename, model in (
        ("classifier_schema.json", TurnClassification),
        ("assessor_schema.json", HazardAssessment),
        ("defence_companion_schema.json", DEFENCE_COMPANION.schema),
        ("hazard_companion_schema.json", HAZARD_COMPANION.schema),
        ("persons_companion_schema.json", PERSONS_COMPANION.schema),
        ("route_companion_schema.json", ROUTE_COMPANION.schema),
    ):
        (GOLDEN / filename).write_text(
            json.dumps(to_strict_json_schema(model), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (GOLDEN / "classifier_system_prompt.txt").write_text(CLASSIFIER_SYSTEM_PROMPT, encoding="utf-8")
    (GOLDEN / "assessor_system_prompt.txt").write_text(ASSESSOR_SYSTEM_PROMPT, encoding="utf-8")
    print(f"wrote goldens for {len(GOLDEN_CASES)} scene and {len(COMPANION_CASES)} companion configurations to {GOLDEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
