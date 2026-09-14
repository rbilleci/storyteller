#!/usr/bin/env bash
# Jump straight into the seeded scene as Maren, the civilised premade
# (sword-master / bodyguard / diplomat, light armour and shield).
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/play-terminal-launch.sh" --premade civilised "$@"
