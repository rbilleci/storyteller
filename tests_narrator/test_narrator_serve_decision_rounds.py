"""Pin ``scripts/narrator_serve.py``'s own decision-rounds resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import narrator_serve  # noqa: E402


def test_the_real_cli_parser_defaults_decision_rounds_to_unset():
    """Zero CLI arguments: ``--decision-rounds`` parses to ``None``, not a bare 0.

    Drives ``narrator_serve._build_parser()``, the exact parser ``main()`` builds and
    parses, so this cannot drift from what a real launch's argument parsing does.
    """
    parser = narrator_serve._build_parser()
    arguments = parser.parse_args([])
    assert arguments.decision_rounds is None


def test_zero_arguments_and_no_environment_override_yields_at_least_one_round():
    """The exact resolution ``main()`` performs: parse, then ``_resolve_decision_rounds``.
    """
    arguments = narrator_serve._build_parser().parse_args([])
    resolved = narrator_serve._resolve_decision_rounds(arguments.decision_rounds, {})
    assert resolved >= 1
    # scripts/probe_combat_decisions.py's attack scenario is the only recorded live
    # measurement of this setting, at rounds=2, matching NarratorConfig's own default.
    assert resolved == 2


def test_the_environment_variable_still_overrides_the_default():
    arguments = narrator_serve._build_parser().parse_args([])
    resolved = narrator_serve._resolve_decision_rounds(
        arguments.decision_rounds, {"BSH_DECISION_ROUNDS": "5"}
    )
    assert resolved == 5


def test_the_cli_flag_wins_over_both_the_environment_and_the_default():
    arguments = narrator_serve._build_parser().parse_args(["--decision-rounds", "3"])
    assert arguments.decision_rounds == 3
    resolved = narrator_serve._resolve_decision_rounds(
        arguments.decision_rounds, {"BSH_DECISION_ROUNDS": "5"}
    )
    assert resolved == 3


def test_zero_still_reproduces_the_pre_fix_adapter_and_replay_arm():
    """An operator or an adapter/replay harness must still be able to disable rounds."""
    arguments = narrator_serve._build_parser().parse_args([])
    resolved = narrator_serve._resolve_decision_rounds(
        arguments.decision_rounds, {"BSH_DECISION_ROUNDS": "0"}
    )
    assert resolved == 0


def test_the_help_text_names_the_real_default_not_a_stale_one():
    """The auditor separately flagged the help text; pin it against silent drift."""
    parser = narrator_serve._build_parser()
    decision_rounds_action = next(
        action for action in parser._actions if action.dest == "decision_rounds"
    )
    assert "BSH_DECISION_ROUNDS or 2" in decision_rounds_action.help
    assert "BSH_DECISION_ROUNDS or 0" not in decision_rounds_action.help
