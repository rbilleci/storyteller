#!/usr/bin/env bash
# Run the narrator against a disposable campaign copy.
#
# Usage:
#   bash scripts/demo_sandbox.sh [transcript-file] [extra narrator_serve.py args...]
#
# The sandbox copies rules/ and world/ into a temporary directory, creates a
# fresh campaign with two pregenerated characters, and runs the narrator there.
# The repository's own campaign/ directory is never touched.
#
# Two interpreters, by necessity. strands-agents 1.50.2 requires mcp<2.0.0 and this
# project pins mcp[cli]>=2,<3, so the narrator cannot share the project environment.
# NARRATOR_PYTHON names the narrator interpreter. See the Quick start section
# at ../README.md#quick-start for the command that creates it.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
transcript="${1:-$repo_root/scripts/demo-combat.txt}"
shift || true

export PYTHONDONTWRITEBYTECODE=1
narrator_python="${NARRATOR_PYTHON:-$repo_root/.narrator-venv/bin/python}"
if [[ ! -x "$narrator_python" ]]; then
  echo "error: no narrator interpreter at $narrator_python" >&2
  echo "Set NARRATOR_PYTHON, or create one with the command in README.md" >&2
  exit 1
fi

sandbox="$(mktemp -d -t bsh-sandbox-XXXXXX)"
cp -r "$repo_root/rules" "$sandbox/rules"
cp -r "$repo_root/world" "$sandbox/world"

echo "sandbox: $sandbox"

cd "$repo_root"
uv run python scripts/new_campaign.py --root "$sandbox" --title "Sandbox" --seed-scene >/dev/null

uv run python - "$sandbox" <<'PYTHON'
import sys
sys.path.insert(0, "src")
from bsh_mcp.service import build_service

service = build_service(sys.argv[1])
# The first field is the channel account identifier.
party = [
    ("player-rill", "Rill", "barbarian", ["scout", "hunter", "survivor"], ["hunting bow", "long knife"], "light"),
    ("player-ossa", "Ossa", "civilised", ["street-urchin", "diplomat", "bookworm"], ["duelling dagger"], "none"),
]
for account_id, name, origin, backgrounds, weapons, armour in party:
    result = service.character_create(account_id, name, origin, backgrounds, weapons=weapons, armour=armour)
    if not result["ok"]:
        raise SystemExit(f"character_create failed: {result['message']}")
    sheet = result["sheet"]
    print(f"  {sheet['id']}: {sheet['hp']} hit points, {sheet['weapon_damage']} damage, {sheet['armour']} armour")
PYTHON

# The narrator interpreter holds mcp 1.29.0; the server child needs mcp 2.x, which
# BSH_SERVER_PYTHON supplies. The project environment is that interpreter.
BSH_SERVER_PYTHON="${BSH_SERVER_PYTHON:-$(uv run python -c 'import sys; print(sys.executable)')}" \
  "$narrator_python" scripts/narrator_serve.py --root "$sandbox" --script "$transcript" "$@"
echo "sandbox retained at $sandbox"
