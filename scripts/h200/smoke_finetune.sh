#!/usr/bin/env bash
# End-to-end check of the fine-tuning path on the H200 with the fast profile: 8 train prompts, two
# per family so that val_fraction 0.5 keeps one each (val is split per family), K = 4 (K = 2 often
# left no pair), provisional scorer, 5 training steps, then one video
# generated with the trained LoRA. The val kl_proxy must move off 0: at 0 the adapter never changed
# the prediction.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"

STEPS=5
RUN_ID=smoke
SMOKE_CAMPAIGN=wan21_smoke

pv_fast() {
    local profile_option
    mapfile -t profile_option < <(error_profile_option fast)
    uv run --no-sync physverify --profile fast --device cuda --set "runtime.threads=$THREADS" \
        "${profile_option[@]}" \
        --set finetune.pairs.min_margin=0 --set finetune.pairs.min_motion=0 \
        --set finetune.pairs.sigma_k=0 --set finetune.pairs.require_object_count=false \
        --set finetune.pairs.min_motion_ratio=0 --set finetune.pairs.max_pairs_per_prompt=2 \
        --set finetune.prompts.val_fraction=0.5 \
        --set finetune.prompts.n_train=8 --set finetune.candidates.k_train=4 "$@"
}

# The smoke campaign is disposable: its provisional scores and pairs are wiped so a new scorer
# version re-scores instead of tripping the stale-stamp guard. checks/*.jsonl live in the score
# directory, so the verifier runs again; extraction outputs stay in the extraction cache.
wipe_provisional() {
    local first campaign root
    first="$(pv_fast finetune status)"
    first="${first%%$'\n'*}"
    campaign="$(sed -n 's/^campaign \([^:]*\):.*/\1/p' <<<"$first")"
    root="$(sed -n 's/.* prompts at //p' <<<"$first")"
    [[ "$campaign" == "$SMOKE_CAMPAIGN" ]] \
        || die "refusing to wipe scores: status reports campaign '$campaign', not $SMOKE_CAMPAIGN"
    [[ -n "$root" && -d "$root" && "${root##*/}" == "$SMOKE_CAMPAIGN" ]] \
        || die "refusing to wipe scores: bad campaign root '$root'"
    rm -rf "$root/scores/base/provisional" "$root/pairs/provisional"
    CAMPAIGN_ROOT="$root"
}

require_cuda
bash scripts/h200/setup.sh
pv_fast finetune prompts build
wipe_provisional
pv_fast finetune generate --split train
pv_fast finetune score --split train --scorer provisional
uv run --no-sync python -c 'import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print("score", r["candidate_id"], r["score"], "failed", r["n_failed"], "of", r["n_decided"])
' "$CAMPAIGN_ROOT/scores/base/provisional/scores.jsonl"
pairs_log="$(mktemp "${TMPDIR:-/tmp}/finetune-smoke-pairs.XXXXXX.log")"
pv_fast finetune pairs --scorer provisional | tee "$pairs_log"
for split in train val; do
    grep -q "^$split: [1-9][0-9]* pairs" "$pairs_log" \
        || die "no $split pairs; see the rejected counts in $pairs_log"
done
log="$(mktemp "${TMPDIR:-/tmp}/finetune-smoke.XXXXXX.log")"
pv_fast finetune train --scorer provisional --allow-provisional --no-resume --force \
    --run-id "$RUN_ID" --max-steps "$STEPS" | tee "$log"
checkpoint="$(sed -n 's/.*last checkpoint //p' "$log" | tail -n 1)"
[[ -n "$checkpoint" && -d "$checkpoint/adapter" ]] \
    || die "training left no checkpoint (see $log)"
kl="$(sed -n 's/.*val implicit_acc .* kl_proxy //p' "$log" | tail -n 1)"
[[ -n "$kl" ]] || die "training reported no val metrics (see $log)"
awk -v kl="$kl" 'BEGIN { exit !(kl + 0 != 0) }' \
    || die "val kl_proxy is $kl after $STEPS steps: the adapter did not change the model"
pv_fast finetune generate --split train --adapter "$checkpoint/adapter" --limit 1
pv_fast finetune status
echo "smoke ok: adapter $checkpoint/adapter"
