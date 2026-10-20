#!/usr/bin/env bash
# One iteration of the perception campaign (ADR 0015 amendment 2 sections 3-4, WP-O) on the H200:
# the error profile, the per-stage sigma and the development metric, all on the kubric_synthetic
# calib clips and written only under its own directory. No call evaluates, abstains, reads real
# video or names an eval clip, so no sealed set is looked at (and PHYSVERIFY_UNSEAL is refused).
#
#   bash scripts/h200/run_iteration.sh [FROM [TO]]   steps FROM..TO (default 1..5); FROM resumes
#
#    1 extract      extract run on the calib clips (failed clips tolerated, noted)
#    2 stages       diagnose stages --clips <calib> --write-profile: the iteration's own error
#                   profile; only calib clips are passed, so the report pass is empty
#    3 relift       extract run again (LIFT reads the new profile)
#    4 stage-sigma  calibrate stage-sigma (D10, calib only)
#    5 devscore     calibrate devscore --level both: devscore.json, and the full calib fit as
#                   <run>/calibration.json (never in a calibration directory); copied to $OUT
#
# Startup (exit 2 before any physverify call): PHYSVERIFY_ITER_ID; HEAD is
# PHYSVERIFY_ITER_EXPECT_COMMIT with nothing uncommitted or untracked under src, configs, scripts
# or annotations; no configs/local.toml and no PHYSVERIFY_CONFIG_DIR; no PHYSVERIFY__* variable
# but the ones exported here (PHYSVERIFY__SEALED__* included); no PHYSVERIFY_UNSEAL or
# PHYSVERIFY_UNSEAL_REPEAT; the output directory (and its calibration/ and runs/) is not inside
# $H200_ROOT/a8v2*, $DATA/calibration*, $DATA/runs* or the baseline $H200_ROOT/kubric-data/
# {calibration,runs}* (whatever DATA is); no other instance on $OUT/run.lock;
# state.env of another device, profile or iteration id; steps 1..FROM-1 completed. Then doctor
# must report config_dir $REPO_ROOT/configs, the device, the profile, the cache, and the
# calibration directory, runs directory, error profile and thresholds inside $OUT.
#
# Required:
#   PHYSVERIFY_ITER_ID              ^it[0-9]+[a-z0-9-]*$, e.g. it0-baseline
#   PHYSVERIFY_ITER_EXPECT_COMMIT   prefix (>= 7 chars) of the HEAD this checkout is at
#   PHYSVERIFY_ITER_CACHE_DIR       paths.cache_dir (normally the shared cache: its keys hold the
#                                   device and provider versions, so an iteration only adds)
# Optional (defaults in brackets):
#   PHYSVERIFY_ITER_PROFILE         sampling profile [precise]; it also fixes the synthetic
#                                   dataset version the calib clips are listed from
#   PHYSVERIFY_ITER_DEVICE          cpu or cuda [cpu]; cuda needs CUDA_VISIBLE_DEVICES in 0-3
#                                   (launch through gpu_queue.sh)
#   PHYSVERIFY_ITER_PYTHON          interpreter [/home/mmonterot/dev/venv/bin/python]
#   PHYSVERIFY_ITER_THREADS         runtime.threads [16: at most 16 while the A8 v2 chain runs]
#   PHYSVERIFY_ITER_DATA            paths.data_dir, the synthetic dataset read only
#                                   [$H200_ROOT/kubric-data]
#   PHYSVERIFY_ITER_DIR             output [$H200_ROOT/iterations/$PHYSVERIFY_ITER_ID-$DEVICE]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steps.sh"
cd "$REPO_ROOT"

FIRST_STEP=1
LAST_STEP=5
[[ $# -le 2 ]] || die "usage: run_iteration.sh [FROM [TO]]"
FROM="${1:-$FIRST_STEP}"
TO="${2:-$LAST_STEP}"
[[ "$FROM" =~ ^[1-5]$ && "$TO" =~ ^[1-5]$ && "$FROM" -le "$TO" ]] \
    || die "FROM and TO must be steps with $FIRST_STEP <= FROM <= TO <= $LAST_STEP, got '$FROM' '$TO'"
PV_PROFILE="${PHYSVERIFY_ITER_PROFILE:-precise}"
[[ "$PV_PROFILE" =~ ^[a-z][a-z0-9_]*$ ]] \
    || die "PHYSVERIFY_ITER_PROFILE must be a profile name, got '$PV_PROFILE'"
KS=(--dataset kubric_synthetic --subset full)
STEP_SAVES=([1]="CALIB_SHA" [2]="STAGES_RUN PROFILE_SHA" [4]="STAGE_SIGMA_SHA"
    [5]="DEVSCORE_SHA CALIBRATION_SHA")

ITER_ID="${PHYSVERIFY_ITER_ID:-}"
[[ "$ITER_ID" =~ ^it[0-9]+[a-z0-9-]*$ ]] \
    || die "PHYSVERIFY_ITER_ID must match ^it[0-9]+[a-z0-9-]*\$ (e.g. it0-baseline), got '$ITER_ID'"

source_h200_env
OWN_VARS=" PATHS__DATA_DIR PATHS__CACHE_DIR PATHS__WEIGHTS_DIR PATHS__VENDOR_DIR PATHS__HF_HOME
    PATHS__CALIBRATION_DIR PATHS__RUNS_DIR RUNTIME__HEAVY_LOCK "
foreign=""
for name in $(compgen -e -X '!PHYSVERIFY__*'); do
    [[ "$OWN_VARS" == *[[:space:]]"${name#PHYSVERIFY__}"[[:space:]]* ]] || foreign+=" $name"
done
[[ -z "$foreign" ]] || die "the environment sets configuration run_iteration.sh does not own:" \
    "$foreign (the iteration would not be the committed one); unset it"
for name in PHYSVERIFY_UNSEAL PHYSVERIFY_UNSEAL_REPEAT; do
    [[ -z "${!name+set}" ]] || die "$name is set: an iteration never looks at a sealed set; unset it"
done
[[ -z "${PHYSVERIFY_CONFIG_DIR+set}" ]] \
    || die "PHYSVERIFY_CONFIG_DIR is set: an iteration reads only $REPO_ROOT/configs; unset it"
[[ ! -e "$REPO_ROOT/configs/local.toml" ]] \
    || die "$REPO_ROOT/configs/local.toml exists: it is not committed; remove it"

EXPECT="${PHYSVERIFY_ITER_EXPECT_COMMIT:-}"
[[ ${#EXPECT} -ge 7 ]] || die "set PHYSVERIFY_ITER_EXPECT_COMMIT to the commit this iteration uses"
HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die "$REPO_ROOT is not a git checkout"
[[ "$HEAD_COMMIT" == "$EXPECT"* ]] || die "HEAD is $HEAD_COMMIT, expected $EXPECT"
DIRTY="$(git -C "$REPO_ROOT" status --porcelain -- src configs scripts annotations)" \
    || die "git status failed in $REPO_ROOT"
[[ -z "$DIRTY" ]] || die "uncommitted or untracked files under src, configs, scripts or" \
    "annotations:"$'\n'"$DIRTY"

DEVICE="${PHYSVERIFY_ITER_DEVICE:-cpu}"
case "$DEVICE" in
    cpu) export CUDA_VISIBLE_DEVICES="" ;;
    cuda)
        [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-3]$ ]] \
            || die "PHYSVERIFY_ITER_DEVICE=cuda needs CUDA_VISIBLE_DEVICES in 0-3 (gpu_queue.sh), got '${CUDA_VISIBLE_DEVICES-unset}'"
        ;;
    *) die "PHYSVERIFY_ITER_DEVICE must be cpu or cuda, got '$DEVICE'" ;;
esac
PV_DEVICE="$DEVICE"
CACHE="${PHYSVERIFY_ITER_CACHE_DIR:-}"
[[ -n "$CACHE" ]] || die "set PHYSVERIFY_ITER_CACHE_DIR to the cache root (normally the shared one)"
PY="${PHYSVERIFY_ITER_PYTHON:-/home/mmonterot/dev/venv/bin/python}"
[[ -x "$PY" ]] || die "no interpreter at $PY (PHYSVERIFY_ITER_PYTHON)"
THREADS="${PHYSVERIFY_ITER_THREADS:-16}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "PHYSVERIFY_ITER_THREADS must be a positive integer"
DATA="${PHYSVERIFY_ITER_DATA:-$H200_ROOT/kubric-data}"
OUT="${PHYSVERIFY_ITER_DIR:-$H200_ROOT/iterations/$ITER_ID-$DEVICE}"

# near PATH BASE: PATH is a directory named BASE* (BASE, BASE-cuda, ...) or below one.
near() {
    [[ "$(realpath -m "$1")" == "$(realpath -m "$2")"* ]]
}

# under PATH DIR: PATH is strictly below DIR.
under() {
    [[ "$(realpath -m "$1")" == "$(realpath -m "$2")"/* ]]
}
for dir in "$OUT" "$OUT/calibration" "$OUT/runs"; do
    for base in "$H200_ROOT/a8v2" "$DATA/calibration" "$DATA/runs" \
        "$H200_ROOT/kubric-data/calibration" "$H200_ROOT/kubric-data/runs"; do
        ! near "$dir" "$base" || die "$dir is inside $base*: an iteration never writes there"
    done
done

mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || die "another run_iteration.sh holds $OUT/run.lock"

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
SUMMARY_TITLE="Iteration $ITER_ID ($DEVICE)"
SUMMARY_INFO="commit $HEAD_COMMIT, profile $PV_PROFILE, device $DEVICE, cache $CACHE, data $DATA, threads $THREADS"
mkdir -p "$LOGS"
touch "$STEPS" "$NOTES"
if [[ -f "$STATE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE"
fi
for pair in "CHAIN_DEVICE $DEVICE" "CHAIN_PROFILE $PV_PROFILE" "CHAIN_ITER_ID $ITER_ID"; do
    read -r key value <<<"$pair"
    if [[ -z "${!key:-}" ]]; then
        save "$key" "$value"
    elif [[ "${!key}" != "$value" ]]; then
        die "$OUT belongs to $key=${!key}, not $value"
    fi
done
require_completed "$FIRST_STEP" "$FROM"

query() {
    env PYTHONPATH="$REPO_ROOT/src" "$PY" "$REPO_ROOT/scripts/h200/a8v2_query.py" "$@"
}

sha_of() {
    sha256sum "$1" | cut -d ' ' -f 1
}

check_doctor() {
    local name
    DOCTOR="$(pv doctor)" || die "physverify doctor failed"
    same_path "$(doctor_value config_dir)" "$REPO_ROOT/configs" \
        || die "doctor reports config_dir '$(doctor_value config_dir)', not $REPO_ROOT/configs"
    [[ "$(doctor_value device)" == "$DEVICE"* ]] \
        || die "doctor reports device '$(doctor_value device)', not $DEVICE"
    [[ "$(doctor_value profile)" == "$PV_PROFILE" ]] \
        || die "doctor reports profile '$(doctor_value profile)', not $PV_PROFILE"
    same_path "$(doctor_value paths.cache_dir)" "$CACHE" \
        || die "doctor reports paths.cache_dir '$(doctor_value paths.cache_dir)', not $CACHE"
    for name in calibration_dir runs_dir error_profile thresholds; do
        under "$(doctor_value "paths.$name")" "$OUT" \
            || die "doctor reports paths.$name '$(doctor_value "paths.$name")', outside $OUT"
    done
    ERROR_PROFILE="$(doctor_value paths.error_profile)"
}

# The calib clips of the synthetic dataset, listed once; a resumed run must see the same list.
calib_clips() {
    local file="$OUT/calib_clips.txt"
    if [[ ! -f "$file" ]]; then
        query --profile "$PV_PROFILE" clips kubric_synthetic full --split calib >"$file.tmp" \
            || die "cannot list the kubric_synthetic calib clips"
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
}

step_1() {
    call 1 -- extract run "${KS[@]}" "${CLIP_ARGS[@]}"
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
    call 1 -- extract run "${KS[@]}" "${CLIP_ARGS[@]}"
}

step_4() {
    call 0 -- calibrate stage-sigma "${KS[@]}"
    save STAGE_SIGMA_SHA "$(sha_of "$OUT/calibration/stage_sigma.json")"
}

step_5() {
    call 0 -- calibrate devscore "${KS[@]}" --level both --out "$OUT/devscore.json"
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
    note "stopped after step $TO as requested; resume with run_iteration.sh $((TO + 1))"
fi
echo "iteration $ITER_ID: $SUMMARY"
