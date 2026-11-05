#!/usr/bin/env bash
# Trains the learned R1 and R2 variants on subset train_h (built by run_train_h.sh).
#
#   bash scripts/h200/run_learned_h.sh OUT EVALUATED_MANIFEST...
#
# Steps (tools/train_learned.py): warm fills the chain cache, each clip under the substitution arm
# it draws, and is the only step that may run a perception model (on PHYSVERIFY_LEARNED_DEVICE,
# cpu by default); images builds the pair image table OUT/pairs.npz from that cache; then samples
# and train run for R1 and R2 into OUT/<check>/. Training refuses an evaluated manifest (calib and
# the B set to be looked at) that shares a perturbation seed with train_h.
# Optional: PHYSVERIFY_LEARNED_DRY_RUN=1 stops after a dry images step (OUT/images_plan.json: the
# clips, pairs and arms it would read; no encoder, no samples, no training, no manifest needed);
# PHYSVERIFY_LEARNED_SKIP_WARM=1 reads a cache filled before.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_train_h.sh"
source_h200_env
cd "$REPO_ROOT"

(($# >= 1)) || die "usage: run_learned_h.sh OUT EVALUATED_MANIFEST..."
OUT="$1"
shift
DRY="${PHYSVERIFY_LEARNED_DRY_RUN:-0}"
[[ "$DRY" == 1 ]] && PHYSVERIFY_LEARNED_DEVICE=cpu
PHYSVERIFY_LEARNED_DEVICE="${PHYSVERIFY_LEARNED_DEVICE:-cpu}"
use_device PHYSVERIFY_LEARNED_DEVICE
[[ "$DRY" == 1 ]] || (($# >= 1)) \
    || die "training needs the evaluated manifests the leakage guard checks"
export_train_h_dataset

SELECT=(--dataset kubric_synthetic --subset train_h)
learned() {
    PYTHONPATH=src uv run --no-sync python tools/train_learned.py \
        --profile "${PHYSVERIFY_KUBRIC_PROFILE:-precise}" --set "runtime.device=$PV_DEVICE" "$@"
}

mkdir -p "$OUT"
if [[ "$DRY" == 1 ]]; then
    learned images "${SELECT[@]}" --out "$OUT/images_plan.json" --dry-run
    exit 0
fi
[[ "${PHYSVERIFY_LEARNED_SKIP_WARM:-0}" == 1 ]] || learned warm "${SELECT[@]}"
learned images "${SELECT[@]}" --out "$OUT/pairs.npz"
ENCODER_SHA="$(PYTHONPATH=src uv run --no-sync python -c \
    "import sys; from pathlib import Path; from physverify.core.feature_table import FeatureTable; \
print(FeatureTable.load(Path(sys.argv[1])).meta['encoder_sha256'])" "$OUT/pairs.npz")" \
    || die "could not read the encoder of $OUT/pairs.npz"
for check in R1 R2; do
    learned samples --check "$check" "${SELECT[@]}" --images "$OUT/pairs.npz" \
        --out "$OUT/$check/samples.npz"
    learned train --check "$check" "${SELECT[@]}" --samples "$OUT/$check/samples.npz" \
        --encoder-sha "$ENCODER_SHA" --evaluated "$@" --out "$OUT/$check"
done
