"""Release-content hygiene checks that work without Git or private source data."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIRS = ("src", "scripts", "tests", "tests_narrator", "docs")
PRIVATE_REFERENCE = re.compile(
    r"\btranscript[-_ ]0\d\b|\bterminal-(?!20000101T000000)\d{8}T\d{6}\b"
    r"|\b(?:prefix|postfix)-\d+\.json\b"
    r"|docs/(?:decision-[\w-]+|PLAN|acceptance-status|remediation-plan[\w-]*)\.md"
)


def test_source_contains_no_session_ids_or_removed_journal_references():
    for directory in CONTENT_DIRS:
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".txt"}:
                continue
            assert not PRIVATE_REFERENCE.search(path.read_text()), path.relative_to(ROOT)


def test_release_documents_contain_no_captured_session_files():
    for path in (ROOT / "docs").rglob("*"):
        if path.is_file():
            assert path.suffix not in {".jsonl", ".log", ".txt"}, path.relative_to(ROOT)


def test_regression_corpus_has_no_source_session_line_metadata():
    source = ROOT / "tests_narrator" / "regression_corpus.py"
    tree = ast.parse(source.read_text())
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REGRESSION_CASES"
                for target in node.targets)
    )
    cases = ast.literal_eval(assignment.value)
    assert cases
    assert all(set(case) == {"text", "route", "hazard", "scope", "note"} for case in cases)
