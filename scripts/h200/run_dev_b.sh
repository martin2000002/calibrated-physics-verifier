#!/usr/bin/env bash
# The B reading of a perception iteration on the development sets of calibration.development_sets,
# with the committed configuration and the iteration's arm only (its CHAIN_ARM swaps from
# perception_arms.tsv go as --set to every call, as in the iteration). Development sets are read as calib by the calib-only
# commands and are not sealed looks, so nothing here unseals or relabels anything.
#
#   bash scripts/h200/run_dev_b.sh [FROM [TO]]   steps FROM..TO (default 1..6); FROM resumes
#
#  kubric_fresh at its declared development version (--set kubric_fresh.version), the clips of
#  PHYSVERIFY_DEV_B_CLIPS_FILE, its own profile:
#    1 fresh-stages    diagnose stages --write-profile (the reading's own error profile)
#    2 fresh-relift    extract run --strict-profile (LIFT reads that profile; a missing or
#                      stale one stops the step)
#    3 fresh-sigma     calibrate stage-sigma
#    4 fresh-devscore  calibrate dev-score --level both: $OUT/fresh/devscore.json
#  kubric_synthetic at its declared development version (--set dataset.version), the
#  iteration's error profile and stage sigma, read from its calibration/ and never written:
#    5 hdri-extract    extract run --strict-profile on calib_clean
#    6 hdri-evaluate   calibrate evaluate --artifact <iteration>/calibration.json --no-real
#                      --score-all-clips: every clean clip scored with the iteration's fit
#
# The dev-score runs of A (the iteration) and of step 4 hold different clips, so they are set
# side by side in the summary, not compared clip by clip (compare-devscore pairs clips).
#
# Startup (exit 2 before any step): the iteration directory completed steps 1-5 of
# run_perception_iteration.sh and its calibration.json and error profile are the ones its
# state.env records; the device (CUDA_VISIBLE_DEVICES in 0-3 for cuda) and profile are the
# iteration's; HEAD is PHYSVERIFY_DEV_B_EXPECT_COMMIT with nothing uncommitted or untracked under
# src, configs, scripts, annotations, pyproject.toml or uv.lock; no configs/local.toml and no
# PHYSVERIFY_CONFIG_DIR; no PHYSVERIFY__* variable but the ones exported here; no PHYSVERIFY_UNSEAL
# or PHYSVERIFY_UNSEAL_REPEAT; the clips file lists clip ids; calibration.development_sets declares
# PHYSVERIFY_DEV_B_FRESH_VERSION of kubric_fresh (or only one version when it is unset) and one
# version of kubric_synthetic; the output directory is not inside the
# calibration chain's directory or the baseline's calibration or runs; no other instance on
# $OUT/run.lock; state.env of another iteration or clip list; steps 1..FROM-1 completed. Then
# doctor, for each set, reports config_dir $REPO_ROOT/configs, the device, the profile, the cache,
# and the set's calibration directory, runs directory and error profile.
#
# Required:
#   PHYSVERIFY_DEV_B_ITERATION_DIR   the run_perception_iteration.sh directory read (A)
#   PHYSVERIFY_DEV_B_EXPECT_COMMIT   prefix (>= 7 chars) of the HEAD this checkout is at
#   PHYSVERIFY_DEV_B_CACHE_DIR       paths.cache_dir (keys hold each set's identity)
#   PHYSVERIFY_DEV_B_CLIPS_FILE      kubric_fresh clip ids read, one per line
# Optional (defaults in brackets):
#   PHYSVERIFY_DEV_B_DRY_RUN         1: the startup checks and doctor, then the commands printed;
#                                    nothing else runs and nothing is written [0]
#   PHYSVERIFY_DEV_B_PYTHON          interpreter [/home/mmonterot/dev/venv/bin/python]
#   PHYSVERIFY_DEV_B_THREADS         runtime.threads [16]
#   PHYSVERIFY_DEV_B_DATA            paths.data_dir, read only [$H200_ROOT/kubric-data]
#   PHYSVERIFY_DEV_B_FRESH_VERSION   the declared kubric_fresh version read; required when
#                                    several are declared [the only one declared]
#   PHYSVERIFY_DEV_B_DIR             output [<iteration>/dev-b]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steps.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_guards.sh"
cd "$REPO_ROOT"

FIRST_STEP=1
LAST_STEP=6
[[ $# -le 2 ]] || die "usage: run_dev_b.sh [FROM [TO]]"
FROM="${1:-$FIRST_STEP}"
TO="${2:-$LAST_STEP}"
[[ "$FROM" =~ ^[1-6]$ && "$TO" =~ ^[1-6]$ && "$FROM" -le "$TO" ]] \
    || die "FROM and TO must be steps with $FIRST_STEP <= FROM <= TO <= $LAST_STEP, got '$FROM' '$TO'"
STEP_SAVES=([1]="FRESH_PROFILE_SHA" [3]="FRESH_SIGMA_SHA" [4]="FRESH_DEVSCORE_SHA"
    [6]="EVALUATION_SHA")
STEP_NAMES=([1]=fresh-stages [2]=fresh-relift [3]=fresh-sigma [4]=fresh-devscore [5]=hdri-extract
    [6]=hdri-evaluate)
ITERATION_STEPS=5
DRY="${PHYSVERIFY_DEV_B_DRY_RUN:-0}"
[[ "$DRY" =~ ^[01]$ ]] || die "PHYSVERIFY_DEV_B_DRY_RUN must be 0 or 1, got '$DRY'"

source_h200_env
refuse_foreign_config run_dev_b.sh " PATHS__DATA_DIR PATHS__CACHE_DIR PATHS__WEIGHTS_DIR
    PATHS__VENDOR_DIR PATHS__HF_HOME PATHS__CALIBRATION_DIR PATHS__RUNS_DIR RUNTIME__HEAVY_LOCK "
for name in PHYSVERIFY_UNSEAL PHYSVERIFY_UNSEAL_REPEAT; do
    [[ -z "${!name+set}" ]] || die "$name is set: a development set is never unsealed; unset it"
done
refuse_local_config run_dev_b.sh
require_checkout PHYSVERIFY_DEV_B_EXPECT_COMMIT

ITER="${PHYSVERIFY_DEV_B_ITERATION_DIR:-}"
[[ -n "$ITER" && -f "$ITER/state.env" ]] \
    || die "set PHYSVERIFY_DEV_B_ITERATION_DIR to a run_perception_iteration.sh directory"
ITER="$(realpath -m "$ITER")"
for ((k = 1; k <= ITERATION_STEPS; k++)); do
    [[ "$(last_exit "$k" "$ITER/steps.tsv")" == 0 ]] \
        || die "$ITER has not completed its step $k: the B reading needs the whole iteration"
done
ITER_DEVSCORE_RUN="$(state_value "$ITER/state.env" DEVSCORE_RUN)"
ITER_PROFILE_SHA="$(state_value "$ITER/state.env" PROFILE_SHA)"
ARTIFACT="$ITER/calibration.json"
[[ -f "$ARTIFACT" && "$(sha_of "$ARTIFACT")" == "$(state_value "$ITER/state.env" CALIBRATION_SHA)" ]] \
    || die "$ARTIFACT is missing or is not the one $ITER/state.env records"
[[ -n "$ITER_PROFILE_SHA" ]] || die "$ITER/state.env records no PROFILE_SHA"
DEV_B_DEVICE="$(state_value "$ITER/state.env" CHAIN_DEVICE)"
use_device DEV_B_DEVICE
PV_PROFILE="$(state_value "$ITER/state.env" CHAIN_PROFILE)"
PV_PROFILE="${PV_PROFILE:-$ASSUMED_CHAIN_PROFILE}"
DEV_B_ARM="$(state_value "$ITER/state.env" CHAIN_ARM)"
ARM_SETS=()
if [[ -n "$DEV_B_ARM" && "$DEV_B_ARM" != none ]]; then
    SWAP_LINES="$(arm_swaps "$DEV_B_ARM")" || exit 2
    mapfile -t SWAPS <<<"$SWAP_LINES"
    for swap in "${SWAPS[@]}"; do
        ARM_SETS+=(--set "$swap")
    done
fi

CACHE="${PHYSVERIFY_DEV_B_CACHE_DIR:-}"
[[ -n "$CACHE" ]] || die "set PHYSVERIFY_DEV_B_CACHE_DIR to the cache root"
PY="${PHYSVERIFY_DEV_B_PYTHON:-/home/mmonterot/dev/venv/bin/python}"
[[ -x "$PY" ]] || die "no interpreter at $PY (PHYSVERIFY_DEV_B_PYTHON)"
THREADS="${PHYSVERIFY_DEV_B_THREADS:-16}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "PHYSVERIFY_DEV_B_THREADS must be a positive integer"
DATA="${PHYSVERIFY_DEV_B_DATA:-$H200_ROOT/kubric-data}"
OUT="$(realpath -m "${PHYSVERIFY_DEV_B_DIR:-$ITER/dev-b}")"

CLIPS_FILE="${PHYSVERIFY_DEV_B_CLIPS_FILE:-}"
[[ -n "$CLIPS_FILE" && -s "$CLIPS_FILE" ]] \
    || die "set PHYSVERIFY_DEV_B_CLIPS_FILE to a non-empty list of kubric_fresh clip ids"
CLIP_ARGS=()
while IFS= read -r id; do
    [[ -z "$id" ]] && continue
    [[ "$id" =~ ^[A-Za-z0-9_.-]+$ ]] || die "$CLIPS_FILE: '$id' is not a clip id"
    CLIP_ARGS+=(--clips "$id")
done <"$CLIPS_FILE"
N_CLIPS=$((${#CLIP_ARGS[@]} / 2))
CLIPS_SHA="$(sha_of "$CLIPS_FILE")"

WANTED_FRESH=()
if [[ -n "${PHYSVERIFY_DEV_B_FRESH_VERSION:-}" ]]; then
    [[ "$PHYSVERIFY_DEV_B_FRESH_VERSION" =~ ^[A-Za-z0-9_.-]+$ ]] \
        || die "PHYSVERIFY_DEV_B_FRESH_VERSION '$PHYSVERIFY_DEV_B_FRESH_VERSION' is not a version"
    WANTED_FRESH=("$PHYSVERIFY_DEV_B_FRESH_VERSION")
fi
FRESH_VERSION="$(query --profile "$PV_PROFILE" development-version kubric_fresh \
    "${WANTED_FRESH[@]}")" || exit 2
HDRI_VERSION="$(query --profile "$PV_PROFILE" development-version kubric_synthetic)" || exit 2
FRESH=(--dataset kubric_fresh --subset full "${CLIP_ARGS[@]}")
HDRI=(--dataset kubric_synthetic --subset calib_clean)

for dir in "$OUT" "$OUT/fresh" "$OUT/hdri"; do
    for base in "$CHAIN_ROOT" "$DATA/calibration" "$DATA/runs" \
        "$H200_ROOT/kubric-data/calibration" "$H200_ROOT/kubric-data/runs" "$ITER/calibration"; do
        ! near "$dir" "$base" || die "$dir is inside $base*: the B reading never writes there"
    done
done

export PHYSVERIFY__PATHS__DATA_DIR="$DATA"
export PHYSVERIFY__PATHS__CACHE_DIR="$CACHE"
export PHYSVERIFY__PATHS__WEIGHTS_DIR="$H200_ROOT/Repo/data/weights"
export PHYSVERIFY__PATHS__VENDOR_DIR="$H200_ROOT/Repo/data/vendor"
export PHYSVERIFY__PATHS__HF_HOME="$H200_ROOT/.cache/huggingface"
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$OUT/heavy.lock"
unset PHYSVERIFY_LOCK_HELD

# fresh_set / hdri_set: the paths and the version override of each set's calls, and whether the
# set reads the iteration's error profile.
fresh_set() {
    SET_NAME="kubric_fresh@$FRESH_VERSION"
    export PHYSVERIFY__PATHS__CALIBRATION_DIR="$OUT/fresh/calibration"
    export PHYSVERIFY__PATHS__RUNS_DIR="$OUT/fresh/runs"
    PV_EXTRA=(--set "kubric_fresh.version=$FRESH_VERSION" "${ARM_SETS[@]}")
    ITERATION_PROFILE=0
}

hdri_set() {
    SET_NAME="kubric_synthetic@$HDRI_VERSION"
    export PHYSVERIFY__PATHS__CALIBRATION_DIR="$ITER/calibration"
    export PHYSVERIFY__PATHS__RUNS_DIR="$OUT/hdri/runs"
    PV_EXTRA=(--set "dataset.version=$HDRI_VERSION" "${ARM_SETS[@]}")
    ITERATION_PROFILE=1
}

# check_set_doctor: doctor confirms the current set's paths and overrides.
check_set_doctor() {
    local profile_path i
    DOCTOR="$(pv doctor)" || die "physverify doctor failed"
    same_path "$(doctor_value config_dir)" "$REPO_ROOT/configs" \
        || die "doctor reports config_dir '$(doctor_value config_dir)', not $REPO_ROOT/configs"
    [[ "$(doctor_value device)" == "$PV_DEVICE"* ]] \
        || die "doctor reports device '$(doctor_value device)', not $PV_DEVICE"
    [[ "$(doctor_value profile)" == "$PV_PROFILE" ]] \
        || die "doctor reports profile '$(doctor_value profile)', not $PV_PROFILE"
    same_path "$(doctor_value paths.cache_dir)" "$CACHE" \
        || die "doctor reports paths.cache_dir '$(doctor_value paths.cache_dir)', not $CACHE"
    same_path "$(doctor_value paths.calibration_dir)" "$PHYSVERIFY__PATHS__CALIBRATION_DIR" \
        || die "$SET_NAME: doctor reports paths.calibration_dir" \
            "'$(doctor_value paths.calibration_dir)', not $PHYSVERIFY__PATHS__CALIBRATION_DIR"
    same_path "$(doctor_value paths.runs_dir)" "$PHYSVERIFY__PATHS__RUNS_DIR" \
        || die "$SET_NAME: doctor reports paths.runs_dir '$(doctor_value paths.runs_dir)', not" \
            "$PHYSVERIFY__PATHS__RUNS_DIR"
    profile_path="$(doctor_value paths.error_profile)"
    under "$profile_path" "$PHYSVERIFY__PATHS__CALIBRATION_DIR" \
        || die "$SET_NAME: doctor reports paths.error_profile '$profile_path', outside" \
            "$PHYSVERIFY__PATHS__CALIBRATION_DIR"
    for ((i = 1; i < ${#PV_EXTRA[@]}; i += 2)); do
        grep -qxF "source: --set ${PV_EXTRA[i]}" <<<"$DOCTOR" \
            || die "$SET_NAME: doctor does not list the override --set ${PV_EXTRA[i]}"
    done
    if [[ "$ITERATION_PROFILE" == 1 ]]; then
        [[ -f "$profile_path" && "$(sha_of "$profile_path")" == "$ITER_PROFILE_SHA" ]] \
            || die "$SET_NAME: $profile_path is not the error profile $ITER/state.env records"
    fi
    SET_PROFILE="$profile_path"
}

hdri_set
check_set_doctor
fresh_set
check_set_doctor
FRESH_PROFILE="$SET_PROFILE"

if [[ "$DRY" == 1 ]]; then
    # call: the arguments pv would pass, printed, never run.
    call() {
        shift 2
        local args=(--profile "$PV_PROFILE" --device "$PV_DEVICE" --set "runtime.threads=$THREADS")
        printf 'dry run (%s): physverify %s\n' "$SET_NAME" "${args[*]} ${PV_EXTRA[*]}${PV_EXTRA[*]:+ }$*"
    }
else
    mkdir -p "$OUT/fresh" "$OUT/hdri"
    exec 9>"$OUT/run.lock"
    flock -n 9 || die "another run_dev_b.sh holds $OUT/run.lock"
fi

LOGS="$OUT/logs"
STATE="$OUT/state.env"
STEPS="$OUT/steps.tsv"
NOTES="$OUT/notes.md"
SUMMARY="$OUT/summary.md"
SUMMARY_TITLE="B reading of $(basename "$ITER") ($PV_DEVICE)"
SUMMARY_INFO="commit $HEAD_COMMIT, iteration $ITER, kubric_fresh@$FRESH_VERSION ($N_CLIPS clips of $CLIPS_FILE, sha256 $CLIPS_SHA), kubric_synthetic@$HDRI_VERSION calib_clean, profile $PV_PROFILE, arm ${DEV_B_ARM:-none} (${ARM_SETS[*]:-no swap}), device $PV_DEVICE${CUDA_VISIBLE_DEVICES:+ (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)}, cache $CACHE, data $DATA, threads $THREADS"
if [[ "$DRY" == 0 ]]; then
    mkdir -p "$LOGS"
    touch "$STEPS" "$NOTES"
    if [[ -f "$STATE" ]]; then
        # shellcheck disable=SC1090
        source "$STATE"
    fi
    for pair in "DEV_B_ITERATION $ITER" "DEV_B_CLIPS_SHA $CLIPS_SHA" \
        "DEV_B_CALIBRATION_SHA $(sha_of "$ARTIFACT")"; do
        read -r key value <<<"$pair"
        if [[ -z "${!key:-}" ]]; then
            save "$key" "$value"
        elif [[ "${!key}" != "$value" ]]; then
            die "$OUT belongs to $key=${!key}, not $value"
        fi
    done
    require_completed "$FIRST_STEP" "$FROM"
fi

# live: false in a dry run, where a step's bookkeeping is skipped.
live() {
    [[ "$DRY" == 0 ]]
}

step_1() {
    call 1 -- diagnose stages "${FRESH[@]}" --write-profile
    live || return 0
    [[ -f "$FRESH_PROFILE" ]] || stop "diagnose stages wrote no $FRESH_PROFILE"
    save FRESH_STAGES_RUN "$(run_of)"
    save FRESH_PROFILE_SHA "$(sha_of "$FRESH_PROFILE")"
    note "$SET_NAME stage errors: $FRESH_STAGES_RUN, profile sha256 $FRESH_PROFILE_SHA"
}

step_2() {
    call 1 -- extract run --strict-profile "${FRESH[@]}"
}

step_3() {
    call 0 -- calibrate stage-sigma "${FRESH[@]}"
    live || return 0
    save FRESH_SIGMA_SHA "$(sha_of "$OUT/fresh/calibration/stage_sigma.json")"
}

step_4() {
    call 0 -- calibrate dev-score "${FRESH[@]}" --level both --out "$OUT/fresh/devscore.json"
    live || return 0
    save FRESH_DEVSCORE_RUN "$(run_of)"
    save FRESH_DEVSCORE_SHA "$(sha_of "$OUT/fresh/devscore.json")"
    { grep -E '^(note: |oracle/|perceived/|  R|selection )' "$LAST_LOG" || true; } \
        | note_block "Development metric B, $SET_NAME, $N_CLIPS clips ($FRESH_DEVSCORE_RUN)"
    note "B devscore.json sha256 $FRESH_DEVSCORE_SHA; A's dev-score run (other clips, side by" \
        "side only): $ITER_DEVSCORE_RUN"
}

step_5() {
    call 1 -- extract run --strict-profile "${HDRI[@]}"
}

step_6() {
    call 0 -- calibrate evaluate --artifact "$ARTIFACT" "${HDRI[@]}" --no-real --score-all-clips
    live || return 0
    local evaluation
    evaluation="$(evaluation_of "$LAST_LOG")"
    save EVALUATION "$evaluation"
    save EVALUATION_SHA "$(sha_of "$evaluation")"
    note "$SET_NAME, every clean clip scored with $ARTIFACT: $evaluation (sha256 $EVALUATION_SHA)"
}

for ((n = FROM; n <= TO; n++)); do
    if [[ "$n" -le 4 ]]; then fresh_set; else hdri_set; fi
    if live; then begin "$n" "${STEP_NAMES[$n]}"; fi
    "step_$n"
    if live; then row 0; fi
done
live || exit 0
if [[ "$TO" -lt "$LAST_STEP" ]]; then
    note "stopped after step $TO as requested; resume with run_dev_b.sh $((TO + 1))"
fi
echo "B reading of $ITER: $SUMMARY"
