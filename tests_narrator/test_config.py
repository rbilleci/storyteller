"""Pin ``NarratorConfig``'s and ``load_config``'s own decision-rounds defaults.

Neither function pins the criterion by itself. ``scripts/narrator_serve.py`` --
the actual production entry point the criterion names -- does not call ``load_config``
anywhere and resolves ``--decision-rounds``/``BSH_DECISION_ROUNDS`` through its own,
independent CLI/env logic; ``load_config`` currently has zero production callers in
this repository. The tests here pin the dataclass default and ``load_config``'s
default as a necessary floor, and as the value a caller that does adopt ``load_config``
would get; they do not by themselves prove what a real zero-argument
``scripts/narrator_serve.py`` launch yields. That pin belongs in a test exercising
``scripts/narrator_serve.py`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from narrator.config import NarratorConfig, load_config  # noqa: E402


def test_the_dataclass_default_runs_at_least_one_decision_round(tmp_path):
    """No caller-supplied value: the dataclass field default alone must be >= 1."""
    config = NarratorConfig(campaign_root=tmp_path)
    assert config.max_decision_rounds >= 1
    assert config.max_decision_rounds == 2


def test_load_config_with_no_environment_override_matches_the_probe_evidence(
    tmp_path, monkeypatch
):
    """``load_config``'s own default, independent of the dataclass field default.

    ``BSH_DECISION_ROUNDS`` must be absent here, so this test proves the value
    ``load_config`` itself resolves rather than the dataclass default alone: the two
    could drift apart if ``load_config`` supplied its own hard-coded fallback. This
    does not exercise ``scripts/narrator_serve.py``, which resolves the same setting
    through separate CLI/env logic of its own -- see the module docstring above.
    """
    monkeypatch.delenv("BSH_DECISION_ROUNDS", raising=False)
    config = load_config(campaign_root=tmp_path)
    assert config.max_decision_rounds >= 1
    # scripts/probe_combat_decisions.py's attack scenario is the only recorded live
    # measurement of this setting, at rounds=2. Matching it is deliberate: diverging
    # would mean shipping a production value the project has never measured live.
    assert config.max_decision_rounds == 2


def test_an_explicit_environment_override_still_wins(tmp_path, monkeypatch):
    """The production default must not swallow an explicit operator override."""
    monkeypatch.setenv("BSH_DECISION_ROUNDS", "5")
    config = load_config(campaign_root=tmp_path)
    assert config.max_decision_rounds == 5


def test_zero_still_reproduces_the_pre_fix_adapter_and_replay_arm(tmp_path, monkeypatch):
    """``BSH_DECISION_ROUNDS=0`` must remain reachable for adapters and transcript replay.

    ``NarratorConfig.max_decision_rounds``'s docstring names this arm explicitly; the
    production default changing must not remove the ability to configure it.
    """
    monkeypatch.setenv("BSH_DECISION_ROUNDS", "0")
    config = load_config(campaign_root=tmp_path)
    assert config.max_decision_rounds == 0


def test_the_gm_address_designator_reads_from_the_environment(tmp_path, monkeypatch):
    """Test the gm address designator reads from the environment.
    """
    import pytest

    from narrator.config import load_config

    monkeypatch.delenv("BSH_GM_ADDRESS", raising=False)
    assert load_config(tmp_path).gm_address == "GM"

    monkeypatch.setenv("BSH_GM_ADDRESS", "MJ")
    assert load_config(tmp_path).gm_address == "MJ"

    for invalid in ("", "  ", "two words", "@GM"):
        monkeypatch.setenv("BSH_GM_ADDRESS", invalid)
        with pytest.raises(ValueError):
            load_config(tmp_path)
