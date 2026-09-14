#!/usr/bin/env bash
# Archive the campaign, world, and rules directories.
#
# Usage:
#   bash scripts/backup_campaign.sh [project-root]
#
# The archive lands in campaign/backups/ and excludes previous backups and the
# campaign lock file, so archives never nest.

set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$root"

if [[ ! -d campaign ]]; then
  echo "error: no campaign directory under $root" >&2
  exit 1
fi

mkdir -p campaign/backups
stamp="$(date -u +%Y%m%d-%H%M%S)"
archive="campaign/backups/campaign-${stamp}.tar.gz"

tar -czf "$archive" \
  --exclude='campaign/backups' \
  --exclude='campaign/.campaign.lock' \
  campaign world rules

echo "$archive"
