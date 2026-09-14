"""Launch the Model Context Protocol server subprocess ``NarratorConfig`` describes.

Two free functions rather than ``NarratorConfig`` methods: ``scripts/narrator_serve.py``'s
preflight check and ``NarratorEngine.start()`` both need this, and neither owns the
other, so this is the shared home rather than a copy kept in either.
"""

from __future__ import annotations

import os

from narrator.config import NarratorConfig


def resolved_server_command(config: NarratorConfig) -> tuple[str, ...]:
    """The argv that launches the MCP server under an interpreter holding mcp 2.x.

    ``BSH_SERVER_PYTHON`` overrides the interpreter, which tests use to point at the
    acceptance environment. Otherwise the launch matches the Hermes profile's own:
    ``uv --directory <root> run python src/bsh_mcp/server.py``.
    """
    if config.server_command:
        return config.server_command
    override = os.environ.get("BSH_SERVER_PYTHON")
    if override:
        return (override, str(config.repo_root / "src" / "bsh_mcp" / "server.py"))
    return (
        "uv",
        "--directory",
        str(config.repo_root),
        "run",
        "python",
        "src/bsh_mcp/server.py",
    )


def server_env(config: NarratorConfig) -> dict[str, str]:
    """Environment for the server child. The campaign root is the only requirement."""
    # PYTHONDONTWRITEBYTECODE is not optional. The Model Context Protocol client
    # inherits only HOME, LOGNAME, PATH, SHELL, TERM and USER by default, so without
    # this the server child writes __pycache__ into the checkout on every run, which
    # a slice worktree forbids and a production run should not do either.
    env = {
        "BSH_CAMPAIGN_ROOT": str(config.campaign_root),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for passthrough in ("BSH_SEED", "BSH_DEBUG", "PATH", "HOME"):
        value = os.environ.get(passthrough)
        if value:
            env[passthrough] = value
    return env
