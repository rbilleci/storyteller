"""Shared configuration for the Strands-dependent test environment."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_bsh_mcp_interpreter() -> str:
    """Resolve the server interpreter in a checkout or extracted source archive."""
    override = os.environ.get("BSH_SERVER_PYTHON")
    if override:
        return override
    return str(REPO_ROOT / ".venv" / "bin" / "python")
