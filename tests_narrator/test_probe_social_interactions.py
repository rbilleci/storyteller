"""Probe artifact tests without a live model endpoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import resolve_bsh_mcp_interpreter  # noqa: E402

INTERPRETER = resolve_bsh_mcp_interpreter()


def test_social_probe_writes_one_redacted_run_bound_report(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "BSH_PROBE_RUN_DIRECTORY": str(tmp_path)}
    result = subprocess.run(
        [INTERPRETER, "scripts/probe_social_interactions.py", "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = tmp_path / "social-production-probe.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["offline"] is True
    assert payload["outputs"] == []
    assert payload["checks"] == {}
    # BLOCKER-4 regression (independent audit, result-1.json): the run directory here
    # has no 40-hex-character parent, so the candidate tree is honestly "unbound"
    # rather than whatever a live `git write-tree` happened to return at run time.
    assert payload["candidate_tree"] == "unbound"
    repeat = subprocess.run(
        [INTERPRETER, "scripts/probe_social_interactions.py", "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    assert repeat.returncode == 75


def test_candidate_tree_reads_the_run_directorys_own_parent_name(tmp_path: Path):
    """BLOCKER-4 regression, at the helper level.
    """
    from probe_social_interactions import _candidate_tree

    candidate = "07a5e1ad4d26a80bd663274d34f272976bc73199"
    bound_run = tmp_path / candidate / "run.abc123"
    bound_run.mkdir(parents=True)
    assert _candidate_tree(bound_run) == candidate

    unbound_run = tmp_path / "not-a-candidate-tree" / "run.abc123"
    unbound_run.mkdir(parents=True)
    assert _candidate_tree(unbound_run) == "unbound"
