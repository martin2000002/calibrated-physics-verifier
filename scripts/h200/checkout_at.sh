#!/usr/bin/env bash
# A clean git checkout of COMMIT of the SOURCE repository in DEST, a new directory, as the
# launchers' require_checkout needs (_guards.sh); SOURCE is only read. COMMIT may be abbreviated:
# git fetch takes only a full sha, so it is resolved in SOURCE first, and an unknown or ambiguous
# prefix stops.
#
#   bash scripts/h200/checkout_at.sh SOURCE COMMIT DEST
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

[[ $# -eq 3 ]] || die "usage: checkout_at.sh SOURCE COMMIT DEST"
SOURCE="$(realpath -e "$1")" || die "no repository at $1"
DEST="$3"
[[ "$2" =~ ^[0-9a-f]{7,40}$ ]] || die "COMMIT must be 7 to 40 hex digits, got '$2'"
COMMIT="$(git -C "$SOURCE" rev-parse --verify "$2^{commit}")" \
    || die "$SOURCE has no single commit $2 (unknown, or an ambiguous prefix: give more digits)"
[[ ! -e "$DEST" ]] || die "$DEST exists: the checkout goes to a new directory"

mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" fetch -q "$SOURCE" "$COMMIT"
git -C "$DEST" reset -q --hard FETCH_HEAD
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] || die "$DEST is not at $COMMIT"
[[ -z "$(git -C "$DEST" status --porcelain)" ]] || die "$DEST is not clean after the checkout"
echo "$DEST at $COMMIT"
