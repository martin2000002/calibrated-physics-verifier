#!/usr/bin/env bash
# Builds the training set of the learned R1 and R2 variants: subset train_h of kubric_synthetic
# at its own version and a seed derived from a fixed label, every scene split=train.
# Before anything is built, every seed it can draw is checked against the manifests of the
# datasets already built (tools/seed_collisions.py); one collision stops it.
#
#   bash scripts/h200/run_train_h.sh [SCENES]   (default: the subset's scenes_per_family)
#
# SCENES above the default extends the same scene stream (indices 0..SCENES-1, same seeds).
# Optional: PHYSVERIFY_TRAIN_H_DRY_RUN=1 runs the seed check only. Everything else as
# run_kubric.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_train_h.sh"
source_h200_env
cd "$REPO_ROOT"

DATA_DIR="${PHYSVERIFY__PATHS__DATA_DIR:-$H200_ROOT/kubric-data}"
export_train_h_dataset
if [[ $# -ge 1 ]]; then
    [[ "$1" =~ ^[1-9][0-9]*$ ]] || die "SCENES must be a positive integer, got '$1'"
    export PHYSVERIFY__DATASET__SUBSETS__TRAIN_H__SCENES_PER_FAMILY="$1"
fi

mapfile -t MANIFESTS < <(find "$DATA_DIR/synthetic" -mindepth 2 -maxdepth 2 -name manifest.jsonl \
    -not -path "*/$TRAIN_H_VERSION/*" | sort)
((${#MANIFESTS[@]} > 0)) || die "no manifest.jsonl under $DATA_DIR/synthetic"
PYTHONPATH=src uv run --no-sync python tools/seed_collisions.py \
    --profile "${PHYSVERIFY_KUBRIC_PROFILE:-precise}" --subset train_h "${MANIFESTS[@]}" \
    || die "seed check failed: the subset would redraw seeds of a built dataset"
[[ "${PHYSVERIFY_TRAIN_H_DRY_RUN:-0}" == 1 ]] && exit 0
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_kubric.sh" train_h
