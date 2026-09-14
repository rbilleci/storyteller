"""Engine-test fixtures shared across tests_narrator/test_engine_*.py."""

from __future__ import annotations

import json
from pathlib import Path

from narrator.config import NarratorConfig
from narrator.engine import NarratorEngine


def _seed_ledger(campaign_root: Path, entries: list[dict]) -> None:
    campaign = campaign_root / "campaign"
    campaign.mkdir(exist_ok=True)
    (campaign / "state.json").write_text(json.dumps({"fiction_debt": entries}), encoding="utf-8")


def _engine_with_stub(campaign_root, stub) -> NarratorEngine:
    """An engine whose agent construction is replaced, so no framework is needed."""
    engine = NarratorEngine(NarratorConfig(campaign_root=campaign_root))
    engine._agent_for = lambda channel_id: stub  # noqa: SLF001 - test seam
    return engine
