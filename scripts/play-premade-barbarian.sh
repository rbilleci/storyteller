#!/usr/bin/env bash
# Jump straight into the seeded scene as Rill, the barbarian premade
# (scout / hunter / survivor, long knife).
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/play-terminal-launch.sh" --premade barbarian "$@"
