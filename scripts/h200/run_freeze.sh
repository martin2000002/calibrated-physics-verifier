#!/usr/bin/env bash
# The freeze on the H200, in a clean checkout at the freeze commit: the chosen perception
# iteration's artifacts adopted as they are (never refitted), the development real video
# extracted, the rule 1.5 evaluation on the calib clips, both R2 detector arms, the exclusion
# record, the abstention and freeze.json. Every call is --profile precise --device cpu; nothing
# names an eval clip or a sealed set, so no look is made (and PHYSVERIFY_UNSEAL is refused).
#
#   bash scripts/h200/run_freeze.sh [FROM [TO]]   steps FROM..TO (default 1..7); FROM resumes
#
#    1 adopt           the selection rule over PHYSVERIFY_FREEZE_ITERATIONS (freeze_query select):
#                      the choice must be PHYSVERIFY_FREEZE_ITERATION and the j3 diff (checks,
#                      physics, extraction, verification) from its commit empty; its
#                      calibration.json, stage_errors.precise.json and stage_sigma.json are copied
#                      to $F/calibration (a different file already there stops)
#    2 real-dev        extract run on physics_iq and pisa_real (full) and the kubric_synthetic calib
#                      clips (LIFT reads the adopted profile; failed clips tolerated, noted)
#    3 evaluate-calib  calibrate evaluate --dataset kubric_synthetic --subset calib --real; stops on
#                      a skipped real dataset, a real set 1.5 does not allow or no primary view
#    4 r2-arms         the R2 clips verified with the base and the tiny detector (rule 1.2)
#    5 decide          exclusions decide, then exclusions show (the record's sha256)
#    6 abstain         selective-risk fit (always; the curve reads eval clips and is not run)
#    7 manifest        freeze_query manifest: $F/freeze.json
#
# Startup (exit 2 before any physverify call): HEAD is PHYSVERIFY_FREEZE_EXPECT_COMMIT with nothing
# uncommitted or untracked under src, configs, scripts, annotations, pyproject.toml or uv.lock; no
# configs/local.toml and no
# PHYSVERIFY_CONFIG_DIR; no PHYSVERIFY__* variable but the ones exported here; no PHYSVERIFY_UNSEAL
# or PHYSVERIFY_UNSEAL_REPEAT; $F (and its calibration/ and runs/) not inside the calibration
# chain's directory, $H200_ROOT/iterations, $DATA/calibration* or $DATA/runs*; no other instance
# on its run.lock; state.env of another commit; steps 1..FROM-1 completed. Then doctor must
# report config_dir $REPO_ROOT/configs, device cpu, profile precise, the cache, and the
# calibration directory, runs directory, error profile and thresholds inside $F.
#
# Required:
#   PHYSVERIFY_FREEZE_EXPECT_COMMIT  prefix (>= 7 chars) of the freeze commit, the HEAD this
#                                    checkout is at
#   PHYSVERIFY_FREEZE_ITERATION      the iteration id the selection rule chooses, written down
#                                    before the run
#   PHYSVERIFY_FREEZE_ITERATIONS     the eligible iteration directories
#                                    (run_perception_iteration.sh), spaces
#   PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT
#                                    the early-extraction commit (the j13 diff starts there)
#   PHYSVERIFY_FREEZE_V1_COMMIT      the v1 commit, the v1 arm's (40 hex)
#   PHYSVERIFY_FREEZE_V1_CALIBRATION the base iteration's calibration.json (the v1 arm's)
#   PHYSVERIFY_FREEZE_CACHE_DIR      paths.cache_dir (the shared cache)
# Optional (defaults in brackets):
#   PHYSVERIFY_FREEZE_DIR            $F [$H200_ROOT/final]
#   PHYSVERIFY_FREEZE_PYTHON         interpreter [/home/mmonterot/dev/venv/bin/python]
#   PHYSVERIFY_FREEZE_THREADS        runtime.threads [32]
#   PHYSVERIFY_FREEZE_DATA           paths.data_dir [$H200_ROOT/kubric-data]
#   PHYSVERIFY_FREEZE_DECIDE_REPLACE 1 to supersede a different decision record (a rerun from 3)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steps.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_guards.sh"
cd "$REPO_ROOT"

FIRST_STEP=1
LAST_STEP=7
[[ $# -le 2 ]] || die "usage: run_freeze.sh [FROM [TO]]"
FROM="${1:-$FIRST_STEP}"
TO="${2:-$LAST_STEP}"
[[ "$FROM" =~ ^[1-7]$ && "$TO" =~ ^[1-7]$ && "$FROM" -le "$TO" ]] \
    || die "FROM and TO must be steps with $FIRST_STEP <= FROM <= TO <= $LAST_STEP, got '$FROM' '$TO'"
KS=(--dataset kubric_synthetic --subset full)
STEP_SAVES=([1]="ADOPTED CALIBRATION_SHA" [3]="EVAL_JSON" [4]="R2_BASE_RUN R2_TINY_RUN"
    [5]="RECORD_SHA" [7]="FREEZE_SHA")
ADOPTED_FILES=(calibration.json calibration/stage_errors.precise.json calibration/stage_sigma.json)

source_h200_env
refuse_foreign_config run_freeze.sh " PATHS__DATA_DIR PATHS__CACHE_DIR PATHS__WEIGHTS_DIR
    PATHS__VENDOR_DIR PATHS__HF_HOME PATHS__CALIBRATION_DIR PATHS__RUNS_DIR RUNTIME__HEAVY_LOCK "
for name in PHYSVERIFY_UNSEAL PHYSVERIFY_UNSEAL_REPEAT; do
    [[ -z "${!name+set}" ]] || die "$name is set: the freeze never looks at a sealed set; unset it"
done
refuse_local_config run_freeze.sh
require_checkout PHYSVERIFY_FREEZE_EXPECT_COMMIT

for name in ITERATION ITERATIONS EARLY_EXTRACTION_COMMIT V1_COMMIT V1_CALIBRATION CACHE_DIR; do
    var="PHYSVERIFY_FREEZE_$name"
    [[ -n "${!var:-}" ]] || die "set $var (see the header of run_freeze.sh)"
done
read -r -a ITERATIONS <<<"$PHYSVERIFY_FREEZE_ITERATIONS"
[[ -f "$PHYSVERIFY_FREEZE_V1_CALIBRATION" ]] \
    || die "no v1 calibration at $PHYSVERIFY_FREEZE_V1_CALIBRATION (PHYSVERIFY_FREEZE_V1_CALIBRATION)"
CACHE="$PHYSVERIFY_FREEZE_CACHE_DIR"
PY="${PHYSVERIFY_FREEZE_PYTHON:-/home/mmonterot/dev/venv/bin/python}"
[[ -x "$PY" ]] || die "no interpreter at $PY (PHYSVERIFY_FREEZE_PYTHON)"
THREADS="${PHYSVERIFY_FREEZE_THREADS:-32}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "PHYSVERIFY_FREEZE_THREADS must be a positive integer"
DATA="${PHYSVERIFY_FREEZE_DATA:-$H200_ROOT/kubric-data}"
F="${PHYSVERIFY_FREEZE_DIR:-$H200_ROOT/final}"
CAL="$F/calibration"
for dir in "$F" "$CAL" "$F/runs"; do
    for base in "$CHAIN_ROOT" "$DATA/calibration" "$DATA/runs" "$H200_ROOT/iterations"; do
        ! near "$dir" "$base" || die "$dir is inside $base*: the freeze never writes there"
    done
done
export CUDA_VISIBLE_DEVICES=""
PV_DEVICE=cpu

OUT="$F/run-freeze"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || die "another run_freeze.sh holds $OUT/run.lock"

export PHYSVERIFY__PATHS__DATA_DIR="$DATA"
export PHYSVERIFY__PATHS__CACHE_DIR="$CACHE"
export PHYSVERIFY__PATHS__WEIGHTS_DIR="$H200_ROOT/Repo/data/weights"
export PHYSVERIFY__PATHS__VENDOR_DIR="$H200_ROOT/Repo/data/vendor"
export PHYSVERIFY__PATHS__HF_HOME="$H200_ROOT/.cache/huggingface"
export PHYSVERIFY__PATHS__CALIBRATION_DIR="$CAL"
export PHYSVERIFY__PATHS__RUNS_DIR="$F/runs"
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$F/heavy.lock"
unset PHYSVERIFY_LOCK_HELD

LOGS="$OUT/logs"
STATE="$OUT/state.env"
STEPS="$OUT/steps.tsv"
NOTES="$OUT/notes.md"
SUMMARY="$OUT/summary.md"
SUMMARY_TITLE="Freeze ($PHYSVERIFY_FREEZE_ITERATION)"
SUMMARY_INFO="commit $HEAD_COMMIT, profile precise, device cpu, cache $CACHE, data $DATA, threads $THREADS"
mkdir -p "$LOGS"
touch "$STEPS" "$NOTES"
if [[ -f "$STATE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE"
fi
for pair in "FREEZE_COMMIT $HEAD_COMMIT" "FREEZE_ITERATION $PHYSVERIFY_FREEZE_ITERATION"; do
    read -r key value <<<"$pair"
    if [[ -z "${!key:-}" ]]; then
        save "$key" "$value"
    elif [[ "${!key}" != "$value" ]]; then
        die "$F belongs to $key=${!key}, not $value (a moved freeze commit repeats the freeze in a new \$F)"
    fi
done
require_completed "$FIRST_STEP" "$FROM"

step_1() {
    local selection="$F/selection.json" chosen dir commit changed file
    fq select "${ITERATIONS[@]}" >"$selection.tmp" || stop "the selection failed (see above)"
    mv "$selection.tmp" "$selection"
    chosen="$(fq chosen "$selection" id)" || stop "no chosen iteration in $selection"
    [[ "$chosen" == "$PHYSVERIFY_FREEZE_ITERATION" ]] \
        || stop "the selection rule chooses $chosen, not PHYSVERIFY_FREEZE_ITERATION=$PHYSVERIFY_FREEZE_ITERATION"
    dir="$(fq chosen "$selection" dir)" || stop "no directory of $chosen in $selection"
    commit="$(fq chosen "$selection" commit)" || stop "no commit of $chosen in $selection"
    changed="$(fq diff j3 "$commit")" || stop "cannot diff the refit paths from $commit"
    [[ -z "$changed" ]] || stop "refit needed: $commit -> HEAD changes checks, physics, extraction or" \
        "verification ($changed): the artifacts cannot be adopted"
    mkdir -p "$CAL"
    for file in "${ADOPTED_FILES[@]}"; do
        copy_once "$dir/$file" "$CAL/$(basename "$file")"
    done
    fq select "${ITERATIONS[@]}" | cmp -s - "$selection" \
        || stop "the iterations changed while they were adopted"
    save ADOPTED "$chosen"
    save CALIBRATION_SHA "$(sha_of "$CAL/calibration.json")"
    note "adopted $chosen ($dir, commit $commit): calibration.json sha256 $CALIBRATION_SHA"
    note_block "Selection ($selection)" <"$selection"
}

step_2() {
    call 1 -- extract run --dataset physics_iq --subset full
    call 1 -- extract run --dataset pisa_real --subset full
    call 1 -- extract run --dataset kubric_synthetic --subset calib
}

step_3() {
    call 0 -- calibrate evaluate --dataset kubric_synthetic --subset calib --real
    local evaluation checked
    evaluation="$(evaluation_of "$LAST_LOG")"
    save EVAL_JSON "$evaluation"
    checked="$(query evaluation "$EVAL_JSON")" || stop "the evaluation failed its checks"
    note_block "Rule 1.5 evaluation on calib ($EVAL_JSON)" <<<"$checked"
}

# Rule 1.2: the same calib clips with the precise detector and the tiny one.
step_4() {
    local clips_file="$F/r2_clips.txt"
    call 0 -- exclusions r2-clips --out "$clips_file"
    local clips=() id base run
    while IFS= read -r id; do
        [[ -z "$id" ]] || clips+=(--clips "$id")
    done <"$clips_file"
    [[ ${#clips[@]} -gt 0 ]] || stop "no R2 clips in $clips_file"
    call 1 -- verify "${KS[@]}" --level perceived --policy raw "${clips[@]}"
    base="$(run_of)"
    save R2_BASE_RUN "$(basename "$base")"
    local tiny=(--set extraction.plan.detect=grounding_dino_tiny)
    call 1 -- "${tiny[@]}" extract run "${KS[@]}" "${clips[@]}"
    call 1 -- "${tiny[@]}" verify "${KS[@]}" --level perceived --policy raw "${clips[@]}"
    run="$(run_of)"
    save R2_TINY_RUN "$(basename "$run")"
    cmp -s "$base/checks_hash.txt" "$run/checks_hash.txt" \
        || stop "the R2 arms ran with different check parameters (checks_hash.txt)"
    note "R2 arms: base $base, tiny $run, checks_hash $(cat "$base/checks_hash.txt")"
}

step_5() {
    local args=(exclusions decide --evaluation "$EVAL_JSON" --r2-base-run "$R2_BASE_RUN"
        --r2-tiny-run "$R2_TINY_RUN")
    [[ "${PHYSVERIFY_FREEZE_DECIDE_REPLACE:-0}" != 1 ]] || args+=(--replace)
    call 0 -- "${args[@]}"
    call 0 -- exclusions show
    local sha
    sha="$(record_sha_of "$LAST_LOG")"
    [[ -n "$sha" ]] || stop "no record sha256 in $LAST_LOG"
    save RECORD_SHA "$sha"
    note "exclusion record sha256 $RECORD_SHA"
}

step_6() {
    call 0 -- selective-risk fit
}

step_7() {
    run_logged 0 -- fq manifest --selection "$F/selection.json" --record-sha "$RECORD_SHA" \
        --p2-commit "$PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT" \
        --v1-commit "$PHYSVERIFY_FREEZE_V1_COMMIT" \
        --v1-calibration "$PHYSVERIFY_FREEZE_V1_CALIBRATION" --out "$F/freeze.json"
    local manifest_log="$LAST_LOG"
    run_logged 0 -- fq artifacts "$F/freeze.json"
    save FREEZE_SHA "$(sha_of "$F/freeze.json")"
    { grep -v '^\$ ' "$manifest_log" || true; } \
        | note_block "Freeze manifest ($F/freeze.json, sha256 $FREEZE_SHA)"
}

check_frozen_doctor "$F" "$CAL" "$CACHE"
for ((n = FROM; n <= TO; n++)); do
    case "$n" in
        1) begin 1 adopt ;;
        2) begin 2 real-dev ;;
        3) begin 3 evaluate-calib ;;
        4) begin 4 r2-arms ;;
        5) begin 5 decide ;;
        6) begin 6 abstain ;;
        7) begin 7 manifest ;;
    esac
    "step_$n"
    row 0
done
if [[ "$TO" -lt "$LAST_STEP" ]]; then
    note "stopped after step $TO as requested; resume with run_freeze.sh $((TO + 1))"
fi
echo "freeze: $SUMMARY"
