"""Campaign validation works without development journals or Git metadata."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_campaign.py"


def test_generated_campaign_validates_without_a_development_journal(service):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(service.store.root), "--json"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["checks"]


def test_missing_campaign_reports_failure_without_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--json"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["ok"] is False
    assert "Traceback" not in result.stderr
