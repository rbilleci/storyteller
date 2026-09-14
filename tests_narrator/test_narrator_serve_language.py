"""Pin ``scripts/narrator_serve.py``'s own language resolution.

A live probe (requested by the user after the table's language, changed to French,
never reached the model) found this file's ``_main`` built ``NarratorConfig`` with no
``language=`` argument at all -- so neither ``--language`` (which did not exist) nor
``BSH_LANGUAGE`` ever reached the engine, the same class of gap
``test_narrator_serve_decision_rounds.py`` independently pinned for decision rounds
(``B1-ROUNDS-DEFAULT-NOT-WIRED``): ``narrator.config.load_config`` is the only reader
of ``BSH_LANGUAGE`` and has no production caller. These tests drive
``scripts/narrator_serve.py``'s real parser and its real resolution function directly,
mirroring that file's structure, so a future edit cannot quietly reintroduce the gap
while every other pin stays green.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import narrator_serve  # noqa: E402


def test_the_real_cli_parser_defaults_language_to_unset():
    parser = narrator_serve._build_parser()
    arguments = parser.parse_args([])
    assert arguments.language is None


def test_zero_arguments_and_no_environment_override_yields_en_us():
    arguments = narrator_serve._build_parser().parse_args([])
    resolved = narrator_serve._resolve_language(arguments.language, {})
    assert resolved == "en-US"


def test_the_environment_variable_still_overrides_the_default():
    arguments = narrator_serve._build_parser().parse_args([])
    resolved = narrator_serve._resolve_language(arguments.language, {"BSH_LANGUAGE": "fr-FR"})
    assert resolved == "fr-FR"


def test_the_cli_flag_wins_over_both_the_environment_and_the_default():
    arguments = narrator_serve._build_parser().parse_args(["--language", "ja-JP"])
    assert arguments.language == "ja-JP"
    resolved = narrator_serve._resolve_language(arguments.language, {"BSH_LANGUAGE": "fr-FR"})
    assert resolved == "ja-JP"
