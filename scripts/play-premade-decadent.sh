#!/usr/bin/env bash
# Jump straight into the seeded scene as Vessa, the decadent premade
# (forbidden-knowledge / snake-blood / vicious, four rolled starting spells).
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/play-terminal-launch.sh" --premade decadent "$@"
