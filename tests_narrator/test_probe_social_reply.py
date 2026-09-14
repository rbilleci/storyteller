"""Social reply probe artifact tests without a live model endpoint.

The probe's value is its live exchange, which cannot run on every validation pass against
a shared graphics processing unit (GPU). What runs here is the probe's own contract, plus
the arming rule that decides whether a green report means anything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import resolve_bsh_mcp_interpreter  # noqa: E402

PROBE = "scripts/probe_social_reply.py"
INTERPRETER = resolve_bsh_mcp_interpreter()


def test_social_reply_probe_writes_one_redacted_run_bound_report(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "BSH_PROBE_RUN_DIRECTORY": str(tmp_path)}
    result = subprocess.run(
        [INTERPRETER, PROBE, "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "social-reply-probe.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["offline"] is True
    assert payload["checks"] == {}
    assert payload["outputs"] == []
    assert "model_id" not in payload

    repeat = subprocess.run(
        [INTERPRETER, PROBE, "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    assert repeat.returncode == 75


def test_the_social_reply_probe_requires_a_run_directory():
    root = Path(__file__).resolve().parents[1]
    environment = {
        key: value for key, value in os.environ.items() if key != "BSH_PROBE_RUN_DIRECTORY"
    }
    result = subprocess.run(
        [INTERPRETER, PROBE, "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 75
    assert "BSH_PROBE_RUN_DIRECTORY" in result.stderr


def test_the_probe_retains_no_narration_in_its_report(tmp_path: Path):
    """The arming test reads narration in process; the report must carry none of it.

    A probe that wrote the model's prose into a retained artifact would publish
    unratified fiction outside the delivery gate. The adapter keeps the text in memory
    and the report keeps only booleans, so this asserts the report's shape directly.
    """
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "BSH_PROBE_RUN_DIRECTORY": str(tmp_path)}
    subprocess.run(
        [INTERPRETER, PROBE, "--offline"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    payload = json.loads((tmp_path / "social-reply-probe.json").read_text(encoding="utf-8"))
    allowed = {
        "status", "candidate_tree", "command_sha256", "launcher_sha256", "checks",
        "outputs", "offline",
    }
    assert set(payload) <= allowed
    for value in payload["checks"].values():
        assert set(value) == {"attempts", "checks"}
        assert all(isinstance(item, bool) for item in value["checks"].values())


def test_the_runtime_mode_refuses_an_incomplete_invocation():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [INTERPRETER, PROBE, "--runtime", "--campaign-root", "/tmp/unused"],
        cwd=root, env={**os.environ}, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
