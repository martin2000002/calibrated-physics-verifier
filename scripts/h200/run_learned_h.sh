#!/usr/bin/env bash
# Trains the learned R1 and R2 variants on subset train_h (built by run_train_h.sh).
#
#   bash scripts/h200/run_learned_h.sh OUT EVALUATED_MANIFEST...
#
# Steps (tools/train_learned.py): warm fills the chain cache, each clip under the substitution arm
# it draws, and also under the all-truth and all-perceived arms the gates read; it is the only step
# that may run a perception model (on PHYSVERIFY_LEARNED_DEVICE, cpu by default). images builds
# the pair image tables from that cache: OUT/pairs.npz (drawn arms), pairs_t1.npz (all truth),
# pairs_t0.npz (all perceived) and pairs_t0_single.npz (all perceived, middle frame alone). Then,
# for R1 and R2 into OUT/<check>/: samples and train on the drawn arms, samples of the three gate
# readings, and gates (gates.json: G-L0, then G-L1 and the controls). Training refuses an
# evaluated manifest (calib and the B set to be looked at) that shares a perturbation seed with
# train_h. C-hdri needs an hdri-dev selection and is run by hand (tools/train_learned.py gates).
# Optional: PHYSVERIFY_LEARNED_DRY_RUN=1 runs no model and computes no value: a dry images step
# (OUT/images_plan.json: the clips, pairs and arms it would read; no encoder), then a plan of
# samples and train per check (OUT/<check>/plan.json: clips by arm, role and validation split,
# the leakage guard against the evaluated manifests given, if any, and the gate and control
# populations; no sample, no training) and the gate populations from the manifests alone
# (OUT/<check>/gate_populations.json; no score);
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
    for check in R1 R2; do
        learned plan --check "$check" "${SELECT[@]}" --out "$OUT/$check/plan.json" \
            --evaluated "$@"
        learned gates --dry-run --check "$check" "${SELECT[@]}" \
            --out "$OUT/$check/gate_populations.json"
    done
    exit 0
fi
if [[ "${PHYSVERIFY_LEARNED_SKIP_WARM:-0}" != 1 ]]; then
    learned warm "${SELECT[@]}"
    learned warm "${SELECT[@]}" --truth-share 1
    learned warm "${SELECT[@]}" --truth-share 0
fi
learned images "${SELECT[@]}" --out "$OUT/pairs.npz"
learned images "${SELECT[@]}" --truth-share 1 --out "$OUT/pairs_t1.npz"
learned images "${SELECT[@]}" --truth-share 0 --out "$OUT/pairs_t0.npz"
learned images "${SELECT[@]}" --truth-share 0 --single-frame --out "$OUT/pairs_t0_single.npz"
ENCODER_SHA="$(PYTHONPATH=src uv run --no-sync python -c \
    "import sys; from pathlib import Path; from physverify.core.feature_table import FeatureTable; \
print(FeatureTable.load(Path(sys.argv[1])).meta['encoder_sha256'])" "$OUT/pairs.npz")" \
    || die "could not read the encoder of $OUT/pairs.npz"
for check in R1 R2; do
    learned samples --check "$check" "${SELECT[@]}" --images "$OUT/pairs.npz" \
        --out "$OUT/$check/samples.npz"
    learned train --check "$check" "${SELECT[@]}" --samples "$OUT/$check/samples.npz" \
        --encoder-sha "$ENCODER_SHA" --evaluated "$@" --out "$OUT/$check"
    for reading in t1:1:pairs_t1 t0:0:pairs_t0 t0_single:0:pairs_t0_single; do
        IFS=: read -r name share table <<<"$reading"
        learned samples --check "$check" "${SELECT[@]}" --truth-share "$share" \
            --images "$OUT/$table.npz" --out "$OUT/$check/samples_$name.npz"
    done
    ARTIFACT_SHA="$(PYTHONPATH=src uv run --no-sync python -c \
        "import json, sys; print(json.load(open(sys.argv[1]))['artifact_sha256'])" \
        "$OUT/$check/training_log.json")" || die "could not read the artifact of $check"
    learned gates --check "$check" "${SELECT[@]}" --artifact "$OUT/$check/$ARTIFACT_SHA.npz" \
        --oracle "$OUT/$check/samples_t1.npz" --perceived "$OUT/$check/samples_t0.npz" \
        --single-frame "$OUT/$check/samples_t0_single.npz" --out "$OUT/$check/gates.json"
done
