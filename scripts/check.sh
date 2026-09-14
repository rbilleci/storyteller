#!/usr/bin/env bash
# Run portable offline acceptance gates from a checkout or extracted source archive.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONDONTWRITEBYTECODE=1
export BSH_SERVER_PYTHON="${BSH_SERVER_PYTHON:-$repo_root/.venv/bin/python}"
export NARRATOR_PYTHON="${NARRATOR_PYTHON:-$repo_root/.narrator-venv/bin/python}"
if [[ ! -x "$BSH_SERVER_PYTHON" ]]; then
  echo "Create the project environment with: uv sync --locked --group dev" >&2
  exit 2
fi
"$BSH_SERVER_PYTHON" -m ruff check .
# Compile in memory: do not leave bytecode in a source release.
"$BSH_SERVER_PYTHON" -c 'from pathlib import Path; [compile(p.read_bytes(), str(p), "exec") for root in ("src", "scripts") for p in Path(root).rglob("*.py")]'
if [[ "${1:-}" == "--fast" ]]; then
  exit 0
fi
if [[ ! -x "$NARRATOR_PYTHON" ]]; then
  echo "Create .narrator-venv using the narrator setup in README.md." >&2
  exit 2
fi
"$BSH_SERVER_PYTHON" -m pytest -q
"$NARRATOR_PYTHON" -m pytest -c pytest-narrator.ini -m "not live" -q
