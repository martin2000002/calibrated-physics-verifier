#!/usr/bin/env bash
# One iteration of the perception campaign on the H200:
# the error profile, the per-stage sigma and the development metric, all on the kubric_synthetic
# calib clips and written only under its own directory. No call evaluates, abstains, reads real
# video or names an eval clip, so no sealed set is looked at (and PHYSVERIFY_UNSEAL is refused).
#
#   bash scripts/h200/run_perception_iteration.sh [FROM [TO]]   steps FROM..TO (default 1..5); FROM resumes
#
#    1 extract      extract run on the calib clips (failed clips tolerated, noted) and on the
#                   calib_f3_extra extension, which dev-score adds for R3b and refuses uncached
#    2 stages       diagnose stages --clips <calib> --write-profile: the iteration's own error
#                   profile; only calib clips are passed, so the report pass is empty
#    3 relift       extract run --strict-profile again on both (LIFT reads the new profile; a
#                   missing or stale profile stops the step)
#    4 stage-sigma  calibrate stage-sigma (per-stage sigma recalibration, calib only)
#    5 devscore     calibrate dev-score --level both: devscore.json, and the full calib fit as
#                   <run>/calibration.json (never in a calibration directory); copied to $OUT
#
# Steps 1-3 may run a model (2 seeds a tracker with the truth; 3 re-runs the camera, whose key holds
# the new profile's focal sigma); 4 and 5 only read the cache. A cuda range FROM >= 4 needs no GPU:
# it keeps the device (the cache keys hold it) and hides every GPU, so a cache miss fails instead of
# using a GPU nobody locked (_common.sh use_step_device). Launch steps 1-3 under the GPU lock and
# 4-5 without it (scripts/h200/README.md, "Iteration launcher"); a cuda range that crosses from 3
# to 4 is refused.
#
# Startup (exit 2 before any physverify call): PHYSVERIFY_ITERATION_ID; HEAD is
# PHYSVERIFY_ITERATION_EXPECT_COMMIT with nothing uncommitted or untracked under src, configs, scripts,
# annotations, pyproject.toml or uv.lock; no configs/local.toml and no PHYSVERIFY_CONFIG_DIR; no PHYSVERIFY__* variable
# but the ones exported here (PHYSVERIFY__SEALED__* included); no PHYSVERIFY_UNSEAL or
# PHYSVERIFY_UNSEAL_REPEAT; the output directory (and its calibration/ and runs/) is not inside
# the calibration chain's directory, $DATA/calibration*, $DATA/runs* or the baseline
# $H200_ROOT/kubric-data/{calibration,runs}* (whatever DATA is); no other instance on $OUT/run.lock;
# state.env of another device, profile or iteration id; steps 1..FROM-1 completed. Then doctor
# must report config_dir $REPO_ROOT/configs, the device, the profile, the cache, and the
# calibration directory, runs directory, error profile and thresholds inside $OUT.
#
# Required:
#   PHYSVERIFY_ITERATION_ID              ^it[0-9]+[a-z0-9-]*$, e.g. it0-baseline
#   PHYSVERIFY_ITERATION_EXPECT_COMMIT   prefix (>= 7 chars) of the HEAD this checkout is at
#   PHYSVERIFY_ITERATION_CACHE_DIR       paths.cache_dir (normally the shared cache: its keys hold the
#                                        device and provider versions, so an iteration only adds)
# Optional (defaults in brackets):
#   PHYSVERIFY_ITERATION_PROFILE         sampling profile [precise]; it also fixes the synthetic
#                                        dataset version the calib clips are listed from
#   PHYSVERIFY_ITERATION_ARM             a declared arm of scripts/h200/perception_arms.tsv [none]:
#                                        its swaps (see the table's header) go as --set to every
#                                        physverify call; doctor must list each one;
#                                        PHYSVERIFY_ITERATION_ID must contain the arm name. The heavy lock is
#                                        per output directory: run arms one at a time
#   PHYSVERIFY_ITERATION_DEVICE          cpu or cuda [cpu]; cuda needs CUDA_VISIBLE_DEVICES in 0-3
#                                        (launch through gpu_queue.sh, the server's GPU queue
#                                        launcher, not in this repository)
#   PHYSVERIFY_ITERATION_PYTHON          interpreter [/home/mmonterot/dev/venv/bin/python]
#   PHYSVERIFY_ITERATION_THREADS         runtime.threads [16: at most 16 while a calibration chain runs]
#   PHYSVERIFY_ITERATION_WORKERS         runtime.workers, processes over clips in steps 2, 4, 5 [1];
#                                        take it from cpu_workers.sh. Outputs do not depend on it.
#                                        Above 1, numpy's BLAS runs one thread per process
#                                        (OMP/MKL/OPENBLAS_NUM_THREADS=1) and the APM injection is
#                                        off, so the forking parent has one thread
#   PHYSVERIFY_ITERATION_DATA            paths.data_dir, the synthetic dataset read only
#                                        [$H200_ROOT/kubric-data]
#   PHYSVERIFY_ITERATION_DIR             output [$H200_ROOT/iterations/$PHYSVERIFY_ITERATION_ID-<device>]
#   PHYSVERIFY_ITERATION_CLIPS_FILE      a rehearsal on some calib clips [all of calib]: one id per
#                                        line, each a calib clip; steps 1-5 pass them as --clips,
#                                        so dev-score and stage-sigma drop the calib_f3_extra
#                                        extension (R3b loses it); the result is not an iteration:
#                                        the freeze and run_dev_b.sh refuse its calibration.json
#   PHYSVERIFY_ITERATION_EXTENSION_LIMIT with CLIPS_FILE only: steps 1 and 3 extract the first N
#                                        calib_f3_extra clips, none with 0 [all of them]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steps.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_guards.sh"
cd "$REPO_ROOT"

FIRST_STEP=1
LAST_STEP=5
GPU_STEPS="1 2 3"
[[ $# -le 2 ]] || die "usage: run_perception_iteration.sh [FROM [TO]]"
FROM="${1:-$FIRST_STEP}"
TO="${2:-$LAST_STEP}"
[[ "$FROM" =~ ^[1-5]$ && "$TO" =~ ^[1-5]$ && "$FROM" -le "$TO" ]] \
    || die "FROM and TO must be steps with $FIRST_STEP <= FROM <= TO <= $LAST_STEP, got '$FROM' '$TO'"
PV_PROFILE="${PHYSVERIFY_ITERATION_PROFILE:-precise}"
[[ "$PV_PROFILE" =~ ^[a-z][a-z0-9_]*$ ]] \
    || die "PHYSVERIFY_ITERATION_PROFILE must be a profile name, got '$PV_PROFILE'"
KS=(--dataset kubric_synthetic --subset full)
EXTENSION=(--dataset kubric_synthetic --subset calib_f3_extra)
STEP_SAVES=([1]="CALIB_SHA" [2]="STAGES_RUN PROFILE_SHA" [4]="STAGE_SIGMA_SHA"
    [5]="DEVSCORE_SHA CALIBRATION_SHA")
ITER_ID="${PHYSVERIFY_ITERATION_ID:-}"
[[ "$ITER_ID" =~ ^it[0-9]+[a-z0-9-]*$ ]] \
    || die "PHYSVERIFY_ITERATION_ID must match ^it[0-9]+[a-z0-9-]*\$ (e.g. it0-baseline), got '$ITER_ID'"

source_h200_env
refuse_foreign_config run_perception_iteration.sh " PATHS__DATA_DIR PATHS__CACHE_DIR PATHS__WEIGHTS_DIR
    PATHS__VENDOR_DIR PATHS__HF_HOME PATHS__CALIBRATION_DIR PATHS__RUNS_DIR RUNTIME__HEAVY_LOCK
    RUNTIME__WORKERS "
for name in PHYSVERIFY_UNSEAL PHYSVERIFY_UNSEAL_REPEAT; do
    [[ -z "${!name+set}" ]] || die "$name is set: an iteration never looks at a sealed set; unset it"
done
refuse_local_config run_perception_iteration.sh
require_checkout PHYSVERIFY_ITERATION_EXPECT_COMMIT

PHYSVERIFY_ITERATION_DEVICE="${PHYSVERIFY_ITERATION_DEVICE:-cpu}"
use_step_device PHYSVERIFY_ITERATION_DEVICE "$FROM" "$TO" "$GPU_STEPS"
ARM="${PHYSVERIFY_ITERATION_ARM:-}"
SWAPS=()
if [[ -n "$ARM" ]]; then
    SWAP_LINES="$(arm_swaps "$ARM")" || exit 2
    [[ "$ITER_ID" == *"$ARM"* ]] \
        || die "PHYSVERIFY_ITERATION_ID '$ITER_ID' must contain the arm name '$ARM' (one directory per arm)"
    mapfile -t SWAPS <<<"$SWAP_LINES"
    for swap in "${SWAPS[@]}"; do
        PV_EXTRA+=(--set "$swap")
    done
fi
CACHE="${PHYSVERIFY_ITERATION_CACHE_DIR:-}"
[[ -n "$CACHE" ]] || die "set PHYSVERIFY_ITERATION_CACHE_DIR to the cache root (normally the shared one)"
PY="${PHYSVERIFY_ITERATION_PYTHON:-/home/mmonterot/dev/venv/bin/python}"
[[ -x "$PY" ]] || die "no interpreter at $PY (PHYSVERIFY_ITERATION_PYTHON)"
THREADS="${PHYSVERIFY_ITERATION_THREADS:-16}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "PHYSVERIFY_ITERATION_THREADS must be a positive integer"
use_workers PHYSVERIFY_ITERATION_WORKERS
DATA="${PHYSVERIFY_ITERATION_DATA:-$H200_ROOT/kubric-data}"
CLIPS_FILE="${PHYSVERIFY_ITERATION_CLIPS_FILE:-}"
[[ -z "$CLIPS_FILE" || -f "$CLIPS_FILE" ]] || die "no clip list at $CLIPS_FILE (PHYSVERIFY_ITERATION_CLIPS_FILE)"
EXTENSION_LIMIT="${PHYSVERIFY_ITERATION_EXTENSION_LIMIT:-}"
[[ -z "$EXTENSION_LIMIT" || -n "$CLIPS_FILE" ]] \
    || die "PHYSVERIFY_ITERATION_EXTENSION_LIMIT needs PHYSVERIFY_ITERATION_CLIPS_FILE (a rehearsal)"
[[ -z "$EXTENSION_LIMIT" || "$EXTENSION_LIMIT" =~ ^[0-9]+$ ]] \
    || die "PHYSVERIFY_ITERATION_EXTENSION_LIMIT must be a count, got '$EXTENSION_LIMIT'"
OUT="${PHYSVERIFY_ITERATION_DIR:-$H200_ROOT/iterations/$ITER_ID-$PV_DEVICE}"

for dir in "$OUT" "$OUT/calibration" "$OUT/runs"; do
    for base in "$CHAIN_ROOT" "$DATA/calibration" "$DATA/runs" \
        "$H200_ROOT/kubric-data/calibration" "$H200_ROOT/kubric-data/runs"; do
        ! near "$dir" "$base" || die "$dir is inside $base*: an iteration never writes there"
    done
done

mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || die "another run_perception_iteration.sh holds $OUT/run.lock"

export PHYSVERIFY__PATHS__DATA_DIR="$DATA"
export PHYSVERIFY__PATHS__CACHE_DIR="$CACHE"
export PHYSVERIFY__PATHS__WEIGHTS_DIR="$H200_ROOT/Repo/data/weights"
export PHYSVERIFY__PATHS__VENDOR_DIR="$H200_ROOT/Repo/data/vendor"
export PHYSVERIFY__PATHS__HF_HOME="$H200_ROOT/.cache/huggingface"
export PHYSVERIFY__PATHS__CALIBRATION_DIR="$OUT/calibration"
export PHYSVERIFY__PATHS__RUNS_DIR="$OUT/runs"
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$OUT/heavy.lock"
unset PHYSVERIFY_LOCK_HELD

LOGS="$OUT/logs"
STATE="$OUT/state.env"
STEPS="$OUT/steps.tsv"
NOTES="$OUT/notes.md"
SUMMARY="$OUT/summary.md"
SUMMARY_TITLE="Iteration $ITER_ID ($PV_DEVICE)"
SUMMARY_INFO="commit $HEAD_COMMIT, profile $PV_PROFILE, arm ${ARM:-none} (${SWAPS[*]:-no swap}), device $PV_DEVICE${CUDA_VISIBLE_DEVICES:+ (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)}, cache $CACHE, data $DATA, threads $THREADS, workers $WORKERS"
mkdir -p "$LOGS"
touch "$STEPS" "$NOTES"
if [[ -f "$STATE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE"
fi
CLIPS_KEY=all
[[ -z "$CLIPS_FILE" ]] || CLIPS_KEY="$(sha_of "$CLIPS_FILE")"
for pair in "CHAIN_DEVICE $PV_DEVICE" "CHAIN_PROFILE $PV_PROFILE" "CHAIN_ITER_ID $ITER_ID" \
    "CHAIN_ARM ${ARM:-none}" "CHAIN_CLIPS $CLIPS_KEY" "CHAIN_EXTENSION ${EXTENSION_LIMIT:-all}"; do
    read -r key value <<<"$pair"
    if [[ -z "${!key:-}" ]]; then
        save "$key" "$value"
    elif [[ "${!key}" != "$value" ]]; then
        die "$OUT belongs to $key=${!key}, not $value"
    fi
done
require_completed "$FIRST_STEP" "$FROM"

check_doctor() {
    local name
    DOCTOR="$(pv doctor)" || die "physverify doctor failed"
    same_path "$(doctor_value config_dir)" "$REPO_ROOT/configs" \
        || die "doctor reports config_dir '$(doctor_value config_dir)', not $REPO_ROOT/configs"
    [[ "$(doctor_value device)" == "$PV_DEVICE"* ]] \
        || die "doctor reports device '$(doctor_value device)', not $PV_DEVICE"
    [[ "$(doctor_value profile)" == "$PV_PROFILE" ]] \
        || die "doctor reports profile '$(doctor_value profile)', not $PV_PROFILE"
    [[ "$(doctor_value workers)" == "$WORKERS" ]] \
        || die "doctor reports workers '$(doctor_value workers)', not $WORKERS"
    same_path "$(doctor_value paths.cache_dir)" "$CACHE" \
        || die "doctor reports paths.cache_dir '$(doctor_value paths.cache_dir)', not $CACHE"
    for name in calibration_dir runs_dir error_profile thresholds; do
        under "$(doctor_value "paths.$name")" "$OUT" \
            || die "doctor reports paths.$name '$(doctor_value "paths.$name")', outside $OUT"
    done
    ERROR_PROFILE="$(doctor_value paths.error_profile)"
    for name in "${SWAPS[@]}"; do
        grep -qxF "source: --set $name" <<<"$DOCTOR" \
            || die "doctor does not list the arm's override --set $name"
    done
}

# The calib clips of the synthetic dataset (those of CLIPS_FILE, in calib order), listed once; a
# resumed run must see the same list.
calib_clips() {
    local file="$OUT/calib_clips.txt" unknown
    if [[ ! -f "$file" ]]; then
        query --profile "$PV_PROFILE" clips kubric_synthetic full --split calib >"$file.tmp" \
            || die "cannot list the kubric_synthetic calib clips"
        if [[ -n "$CLIPS_FILE" ]]; then
            unknown="$(grep -vxF -f "$file.tmp" "$CLIPS_FILE" | grep . || true)"
            [[ -z "$unknown" ]] || die "$CLIPS_FILE lists clips that are not calib clips:"$'\n'"$unknown"
            grep -xF -f "$CLIPS_FILE" "$file.tmp" >"$file.part" || true
            mv "$file.part" "$file.tmp"
        fi
        mv "$file.tmp" "$file"
    fi
    [[ -s "$file" ]] || die "no kubric_synthetic calib clip in $file"
    [[ -z "${CALIB_SHA:-}" || "$(sha_of "$file")" == "$CALIB_SHA" ]] \
        || die "$file changed since step 1"
    CLIP_ARGS=()
    local id
    while IFS= read -r id; do
        [[ -z "$id" ]] || CLIP_ARGS+=(--clips "$id")
    done <"$file"
    FIT_ARGS=()
    [[ -z "$CLIPS_FILE" ]] || FIT_ARGS=("${CLIP_ARGS[@]}")
}

# EXTENSION_ARGS: the calib_f3_extra extraction of steps 1 and 3 (empty with a limit of 0).
extension_clips() {
    EXTENSION_ARGS=("${EXTENSION[@]}")
    [[ -n "$EXTENSION_LIMIT" ]] || return 0
    EXTENSION_ARGS=()
    [[ "$EXTENSION_LIMIT" -gt 0 ]] || return 0
    local file="$OUT/extension_clips.txt" id
    if [[ ! -f "$file" ]]; then
        query --profile "$PV_PROFILE" clips kubric_synthetic calib_f3_extra \
            | sed -n "1,${EXTENSION_LIMIT}p" >"$file.tmp" \
            || die "cannot list the kubric_synthetic calib_f3_extra clips"
        mv "$file.tmp" "$file"
    fi
    EXTENSION_ARGS=("${EXTENSION[@]}")
    while IFS= read -r id; do
        [[ -z "$id" ]] || EXTENSION_ARGS+=(--clips "$id")
    done <"$file"
}

# extract_extension [ARG...]: extract run on the extension unless the limit skips it.
extract_extension() {
    [[ ${#EXTENSION_ARGS[@]} -eq 0 ]] || call 1 -- extract run "$@" "${EXTENSION_ARGS[@]}"
}

step_1() {
    call 1 -- extract run "${KS[@]}" "${CLIP_ARGS[@]}"
    extract_extension
    save CALIB_SHA "$(sha_of "$OUT/calib_clips.txt")"
}

step_2() {
    call 1 -- diagnose stages "${KS[@]}" "${CLIP_ARGS[@]}" --write-profile
    local run
    run="$(run_of)"
    save STAGES_RUN "$run"
    [[ -f "$ERROR_PROFILE" ]] || stop "diagnose stages wrote no $ERROR_PROFILE"
    save PROFILE_SHA "$(sha_of "$ERROR_PROFILE")"
    note "stage errors: $STAGES_RUN, profile $ERROR_PROFILE sha256 $PROFILE_SHA"
}

step_3() {
    call 1 -- extract run --strict-profile "${KS[@]}" "${CLIP_ARGS[@]}"
    extract_extension --strict-profile
}

step_4() {
    call 0 -- calibrate stage-sigma "${KS[@]}" "${FIT_ARGS[@]}"
    save STAGE_SIGMA_SHA "$(sha_of "$OUT/calibration/stage_sigma.json")"
}

step_5() {
    call 0 -- calibrate dev-score "${KS[@]}" "${FIT_ARGS[@]}" --level both --out "$OUT/devscore.json"
    local run
    run="$(run_of)"
    [[ -f "$run/calibration.json" ]] || stop "no full calib fit in $run"
    cp "$run/calibration.json" "$OUT/calibration.json"
    save DEVSCORE_RUN "$run"
    save DEVSCORE_SHA "$(sha_of "$OUT/devscore.json")"
    save CALIBRATION_SHA "$(sha_of "$OUT/calibration.json")"
    { grep -E '^(note: |oracle/|perceived/|  R|selection )' "$LAST_LOG" || true; } \
        | note_block "Development metric ($run)"
    note "devscore.json sha256 $DEVSCORE_SHA; calibration.json (full calib fit) sha256" \
        "$CALIBRATION_SHA; error profile sha256 ${PROFILE_SHA:-?}"
}

check_doctor
calib_clips
extension_clips
[[ -z "$CLIPS_FILE" ]] || note "rehearsal on $(grep -c . "$OUT/calib_clips.txt") calib clips of" \
    "$CLIPS_FILE: steps 4-5 pass --clips, so they drop the calib_f3_extra extension (R3b)"
for ((n = FROM; n <= TO; n++)); do
    case "$n" in
        1) begin 1 extract ;;
        2) begin 2 stages ;;
        3) begin 3 relift ;;
        4) begin 4 stage-sigma ;;
        5) begin 5 devscore ;;
    esac
    "step_$n"
    row 0
done
if [[ "$TO" -lt "$LAST_STEP" ]]; then
    note "stopped after step $TO as requested; resume with run_perception_iteration.sh $((TO + 1))"
fi
echo "iteration $ITER_ID: $SUMMARY"
