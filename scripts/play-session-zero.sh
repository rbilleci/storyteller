#!/usr/bin/env bash
# Start a terminal session at session zero: a fresh disposable campaign with no
# characters and no seeded scene. The narrator runs the bsh-session-zero procedure
# — tone, boundaries, guided character creation through the tools, first scene.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/play-terminal-launch.sh" --session-zero "$@"
