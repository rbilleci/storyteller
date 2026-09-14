#!/usr/bin/env bash
# Shared launcher behind the play-session-zero.sh / play-premade-*.sh entry points.
# Resolves the project interpreter and hands every argument to play_terminal.py,
# which bootstraps a disposable campaign and spawns the narrator terminal channel
# (the narrator interpreter resolves itself from .narrator-venv, or from
# NARRATOR_PYTHON when set).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1

if [[ -x "$root/.venv/bin/python" ]]; then
    exec "$root/.venv/bin/python" "$root/scripts/play_terminal.py" "$@"
fi
if command -v uv >/dev/null 2>&1; then
    cd "$root"
    exec uv run python scripts/play_terminal.py "$@"
fi
echo "error: no project interpreter at $root/.venv and no uv on PATH." >&2
echo "  create one with: uv sync --group dev" >&2
exit 2
