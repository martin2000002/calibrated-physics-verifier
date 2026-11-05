#!/usr/bin/env bash
# Renders an R1_dyn set on the H200 CPU through run_kubric.sh: f5_contact filmed
# along the motion, each scene with its clean clip and one r1_pass_through level cycled by scene
# index (L = index % 4 + 1), as a version of its own under paths.synthetic_dir.
#
#   bash scripts/h200/run_r1dyn_sets.sh calib   # calib-r1dyn: seed 2046980616, every scene calib
#   bash scripts/h200/run_r1dyn_sets.sh b5      # v2-dev-b5: a seed drawn here once, every scene eval
#
# The b5 seed is drawn from /dev/urandom on the first run and kept in <version>/SEED; a resumed run
# reads it back. Before the build, tools/seed_collisions.py checks the dataset, scene and attempt
# seeds against the manifest of every set in R1DYN_SEED_SETS (default: every other directory of
# the synthetic dir with a manifest) and stops on a collision.
#
# Optional: R1DYN_SCENES scenes (default 60; a smoke render takes <= 3 and its own
# PHYSVERIFY__PATHS__DATA_DIR), RESUME=1 to resume a store that has a plan, and run_kubric.sh's
# variables.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SET="${1:-}"
SCENES="${R1DYN_SCENES:-60}"
SUBSET=r1dyn
CALIB_SEED=2046980616
DATA_DIR="${PHYSVERIFY__PATHS__DATA_DIR:-$H200_ROOT/kubric-data}"
SYNTH="$DATA_DIR/synthetic"

case "$SET" in
    calib) VERSION=calib-r1dyn FRACTION=1.0 SPLIT=calib ;;
    b5) VERSION=v2-dev-b5 FRACTION=0.0 SPLIT=eval ;;
    *) die "usage: run_r1dyn_sets.sh calib|b5" ;;
esac
[[ "$SCENES" =~ ^[1-9][0-9]*$ ]] || die "R1DYN_SCENES must be a positive integer, got '$SCENES'"
[[ -n "${PHYSVERIFY_UNSEAL:-}${PHYSVERIFY_UNSEAL_REPEAT:-}" ]] && die "unset PHYSVERIFY_UNSEAL*"
STORE="$SYNTH/$VERSION"
[[ ! -e "$STORE/plan.json" || -n "${RESUME:-}" ]] \
    || die "$STORE has a plan; set RESUME=1 to resume it"

mkdir -p "$STORE"
if [[ "$SET" == calib ]]; then
    SEED="$CALIB_SEED"
elif [[ -s "$STORE/SEED" ]]; then
    SEED="$(<"$STORE/SEED")"
else
    SEED="$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')"
    printf '%s\n' "$SEED" >"$STORE/SEED"
fi
[[ "$SEED" =~ ^[0-9]+$ ]] || die "bad seed '$SEED' in $STORE/SEED"

export PHYSVERIFY__PATHS__DATA_DIR="$DATA_DIR"
export PHYSVERIFY__DATASET__VERSION="$VERSION"
export PHYSVERIFY__DATASET__SEED="$SEED"
export PHYSVERIFY__DATASET__CALIB_FRACTION="$FRACTION"
export PHYSVERIFY__DATASET__FAMILIES__F5_CONTACT__CAMERA_AXIS=along
export PHYSVERIFY__DATASET__SUBSETS__R1DYN="{ scenes_per_family = $SCENES, levels = [1, 2, 3, 4], \
level_mode = \"cycle\", split = \"$SPLIT\", perturbations = { f5_contact = [\"r1_pass_through\"] } }"

if [[ -n "${R1DYN_SEED_SETS:-}" ]]; then
    read -r -a SEED_SETS <<<"$R1DYN_SEED_SETS"
else
    SEED_SETS=()
    for manifest in "$SYNTH"/*/manifest.jsonl; do
        [[ -e "$manifest" && "$(dirname "$manifest")" != "$STORE" ]] \
            && SEED_SETS+=("$(dirname "$manifest")")
    done
fi
if ((${#SEED_SETS[@]})); then
    (cd "$REPO_ROOT" && uv run --no-sync python tools/seed_collisions.py \
        --profile "${PHYSVERIFY_KUBRIC_PROFILE:-precise}" --subset "$SUBSET" "${SEED_SETS[@]}") \
        | tee "$STORE/seedcheck-$(date -u +%Y%m%dT%H%M%SZ).json" \
        || die "seed check failed or found a collision (see $STORE/seedcheck-*.json)"
fi

exec bash "$REPO_ROOT/scripts/h200/run_kubric.sh" "$SUBSET"
