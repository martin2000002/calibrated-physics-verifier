#!/usr/bin/env bash
# H2 (ADR 0011; ADR 0015 Enmienda 1: CPU) on the H200, in two phases a person runs in turn:
#
#   bash scripts/h200/run_h2.sh dev    pre, dev-fetch, dev-extract, dev-verify, freeze (h2_dev);
#                                      ends with exit 0 and the frozen rule, gate and cut in
#                                      summary.md, which a person reads before the test phase
#   bash scripts/h200/run_h2.sh test   test-pre, test-fetch, test-extract, test-verify, then the
#                                      one pre-registered `agreement compare --rule calibrated`
#
# Every call is --profile precise --device cpu (the VideoPhy cache keys are the CPU ones) through
# $PHYSVERIFY_H2_PYTHON, with CUDA_VISIBLE_DEVICES empty: no uv, no setup.sh, no GPU. Extract and
# verify exit 1 when some clips fail; compare counts those clips, so that is not fatal. The `pre`
# steps verify one cached clip and stop when its checks carry an error (an uncached heavy stage),
# so a wrong cache stops before hours of recomputation.
#
# Startup (exit 2, before any physverify call): the phase argument; no PHYSVERIFY__* variable but
# the ones this script exports (a foreign one would change verify or the rule without moving the
# frozen hashes); PHYSVERIFY_H2_DEVICE cpu; HEAD is PHYSVERIFY_H2_EXPECT_COMMIT with nothing
# uncommitted or untracked under src, configs, scripts or annotations; no other instance on
# $OUT/run.lock; the A8 v2 CPU chain in PHYSVERIFY_H2_A8V2_DIR completed steps 0..9 and published
# calibration.json, exclusions.json and stage_errors.precise.json in the calibration directory.
# Then doctor must report device cpu, the four exported paths, the precise error profile and the
# thresholds of that calibration directory, and the caption review must be recorded.
#
# dev refuses once a test comparison is logged (a freeze after looking at test is post hoc). After
# the freeze the pre-registration must be calibrated (or carry a calibration gate), cpu, precise,
# and bound to A8 v2's calibration.json and decision record. state.env keeps what test checks:
# the pre-registration and its sha256, the frozen commit, the sha256 of the A8 v2 files, of the
# exported PHYSVERIFY__* environment and doctor's config_hash.
#
# test runs once: it refuses, with no physverify call, when state.env records TEST_COMPARED or
# the comparison log has an entry (no variable skips this). It requires the freeze step's last
# exit 0 and every value dev recorded unchanged. A pre-registration with a calibration gate (the
# stratum was not usable) makes H2 "not evaluable with the calibrated rule" whether or not test
# runs; test then needs PHYSVERIFY_H2_ACCEPT_NOT_EVALUABLE=1 to spend it on that result.
#
# Optional (defaults in brackets):
#   PHYSVERIFY_H2_EXPECT_COMMIT     required: prefix (>= 7 chars) of the HEAD this checkout is at
#   PHYSVERIFY_H2_PYTHON            interpreter [/home/mmonterot/dev/venv/bin/python]
#   PHYSVERIFY_H2_THREADS           runtime.threads [32]
#   PHYSVERIFY_H2_DIR               output: logs/, steps.tsv, state.env, summary.md [$H200_ROOT/h2]
#   PHYSVERIFY_H2_DATA              paths.data_dir: runs, agreement/ [$H200_ROOT/h2-data]
#   PHYSVERIFY_H2_CACHE_DIR         paths.cache_dir [$H200_ROOT/Repo/data/cache]
#   PHYSVERIFY_H2_RAW_DIR           paths.raw_dir, the VideoPhy videos [$H200_ROOT/Repo/data/raw]
#   PHYSVERIFY_H2_CALIBRATION_DIR   A8 v2 CPU artifacts, read only [$H200_ROOT/kubric-data/calibration]
#   PHYSVERIFY_H2_A8V2_DIR          the A8 v2 CPU run, read only [$H200_ROOT/a8v2]
#   PHYSVERIFY_H2_PREFLIGHT_CLIP    dev preflight clip [v1-a8208f4dd41e]
#   PHYSVERIFY_H2_PREFLIGHT_TEST_CLIP  test preflight clip [the first h2_test clip with a video
#                                   and an object query]
#   PHYSVERIFY_H2_ACCEPT_NOT_EVALUABLE  1 to run test on a pre-registration with a calibration gate
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steps.sh"
cd "$REPO_ROOT"

[[ $# -eq 1 && ("$1" == dev || "$1" == test) ]] || die "usage: run_h2.sh dev|test"
PHASE="$1"
DEV_SUBSET=h2_dev
TEST_SUBSET=h2_test
H2_LINK=h2-latest
A8V2_STEPS=9
CALIBRATED=calibrated
PROVISIONAL=provisional_any_fail

source_h200_env
OWN_VARS=" PATHS__DATA_DIR PATHS__CACHE_DIR PATHS__RAW_DIR PATHS__CALIBRATION_DIR PATHS__WEIGHTS_DIR
    PATHS__VENDOR_DIR PATHS__HF_HOME RUNTIME__HEAVY_LOCK CHECK_SCOPE__REQUIRE_DECISION_RECORD "
foreign=""
for name in $(compgen -e -X '!PHYSVERIFY__*'); do
    [[ "$OWN_VARS" == *[[:space:]]"${name#PHYSVERIFY__}"[[:space:]]* ]] || foreign+=" $name"
done
[[ -z "$foreign" ]] || die "the environment sets configuration run_h2.sh does not own:$foreign" \
    "(it would change H2 without moving its frozen hashes); unset it"

DEVICE="${PHYSVERIFY_H2_DEVICE:-cpu}"
[[ "$DEVICE" == cpu ]] \
    || die "PHYSVERIFY_H2_DEVICE must be cpu (ADR 0015 Enmienda 1; the VideoPhy cache is the CPU one), got '$DEVICE'"
EXPECT="${PHYSVERIFY_H2_EXPECT_COMMIT:-}"
[[ ${#EXPECT} -ge 7 ]] || die "set PHYSVERIFY_H2_EXPECT_COMMIT to the commit this run uses"
HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die "$REPO_ROOT is not a git checkout"
[[ "$HEAD_COMMIT" == "$EXPECT"* ]] || die "HEAD is $HEAD_COMMIT, expected $EXPECT"
DIRTY="$(git -C "$REPO_ROOT" status --porcelain -- src configs scripts annotations)" \
    || die "git status failed in $REPO_ROOT"
[[ -z "$DIRTY" ]] || die "uncommitted or untracked files under src, configs, scripts or" \
    "annotations (they enter the frozen hashes):"$'\n'"$DIRTY"
PY="${PHYSVERIFY_H2_PYTHON:-/home/mmonterot/dev/venv/bin/python}"
[[ -x "$PY" ]] || die "no interpreter at $PY (PHYSVERIFY_H2_PYTHON)"
THREADS="${PHYSVERIFY_H2_THREADS:-32}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "PHYSVERIFY_H2_THREADS must be a positive integer"

OUT="${PHYSVERIFY_H2_DIR:-$H200_ROOT/h2}"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || die "another run_h2.sh holds $OUT/run.lock"

export PHYSVERIFY__PATHS__DATA_DIR="${PHYSVERIFY_H2_DATA:-$H200_ROOT/h2-data}"
export PHYSVERIFY__PATHS__CACHE_DIR="${PHYSVERIFY_H2_CACHE_DIR:-$H200_ROOT/Repo/data/cache}"
export PHYSVERIFY__PATHS__RAW_DIR="${PHYSVERIFY_H2_RAW_DIR:-$H200_ROOT/Repo/data/raw}"
export PHYSVERIFY__PATHS__CALIBRATION_DIR="${PHYSVERIFY_H2_CALIBRATION_DIR:-$H200_ROOT/kubric-data/calibration}"
export PHYSVERIFY__PATHS__WEIGHTS_DIR="$H200_ROOT/Repo/data/weights"
export PHYSVERIFY__PATHS__VENDOR_DIR="$H200_ROOT/Repo/data/vendor"
export PHYSVERIFY__PATHS__HF_HOME="$H200_ROOT/.cache/huggingface"
# The CPU chain's lock: no heavy H2 step runs while an A8 v2 one does.
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$H200_ROOT/kubric.lock"
# ADR 0015 runbook step 10: no H2 step runs without the sealed exclusion record.
export PHYSVERIFY__CHECK_SCOPE__REQUIRE_DECISION_RECORD=true
export CUDA_VISIBLE_DEVICES=""
unset PHYSVERIFY_LOCK_HELD
PV_DEVICE=cpu
CAL="$PHYSVERIFY__PATHS__CALIBRATION_DIR"
A8V2="${PHYSVERIFY_H2_A8V2_DIR:-$H200_ROOT/a8v2}"
ENV_SHA="$(env | grep '^PHYSVERIFY__' | LC_ALL=C sort | sha256sum | cut -d ' ' -f 1)"

LOGS="$OUT/logs"
STATE="$OUT/state.env"
STEPS="$OUT/steps.tsv"
NOTES="$OUT/notes.md"
SUMMARY="$OUT/summary.md"
SUMMARY_TITLE="H2 on VideoPhy (CPU)"
SUMMARY_INFO="commit $HEAD_COMMIT, data $PHYSVERIFY__PATHS__DATA_DIR, cache $PHYSVERIFY__PATHS__CACHE_DIR, calibration $CAL, threads $THREADS"
mkdir -p "$LOGS"
touch "$STEPS" "$NOTES"
if [[ -f "$STATE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE"
fi

query() {
    env PYTHONPATH="$REPO_ROOT/src" "$PY" "$REPO_ROOT/scripts/h200/$1" "${@:2}"
}

sha_of() {
    sha256sum "$1" | cut -d ' ' -f 1
}

# a8v2_state KEY: KEY of the A8 v2 state.env, read in a subshell so nothing here is overwritten.
a8v2_state() {
    (
        # shellcheck disable=SC1091
        [[ -f "$A8V2/state.env" ]] && source "$A8V2/state.env"
        printf '%s' "${!1:-}"
    )
}

require_a8v2() {
    local device k code name
    device="$(a8v2_state CHAIN_DEVICE)"
    [[ "$device" == cpu ]] || die "$A8V2 is not the A8 v2 cpu chain (CHAIN_DEVICE '$device')"
    A8V2_RECORD_SHA="$(a8v2_state RECORD_SHA)"
    [[ -n "$A8V2_RECORD_SHA" ]] || die "$A8V2/state.env has no RECORD_SHA: A8 v2 step 9 is not done"
    for ((k = 0; k <= A8V2_STEPS; k++)); do
        code="$(last_exit "$k" "$A8V2/steps.tsv")"
        [[ "$code" == 0 ]] \
            || die "A8 v2 step $k has not completed (last exit '${code:-none}' in $A8V2/steps.tsv)"
    done
    for name in calibration.json exclusions.json stage_errors.precise.json; do
        [[ -f "$CAL/$name" ]] || die "no $CAL/$name: A8 v2 has not published it"
    done
}

save_artifacts() {
    save CALIB_SHA "$(sha_of "$CAL/calibration.json")"
    save RECORD_FILE_SHA "$(sha_of "$CAL/exclusions.json")"
    save PROFILE_SHA "$(sha_of "$CAL/stage_errors.precise.json")"
}

artifacts_unchanged() {
    [[ "$(sha_of "$CAL/calibration.json")" == "${CALIB_SHA:-}" \
        && "$(sha_of "$CAL/exclusions.json")" == "${RECORD_FILE_SHA:-}" \
        && "$(sha_of "$CAL/stage_errors.precise.json")" == "${PROFILE_SHA:-}" ]] \
        || die "the A8 v2 files in $CAL changed since the dev phase: H2 must be frozen again"
}

# The comparison log of the configured pre-registration path holds no entry.
untested() {
    local n
    n="$(query h2_query.py tested)" || die "cannot read the test-comparison log"
    [[ "$n" == 0 ]]
}

check_doctor() {
    local name var
    DOCTOR="$(pv doctor)" || die "physverify doctor failed"
    [[ "$(doctor_value device)" == cpu ]] || die "doctor reports device '$(doctor_value device)', not cpu"
    for name in DATA_DIR CACHE_DIR RAW_DIR CALIBRATION_DIR; do
        var="PHYSVERIFY__PATHS__$name"
        same_path "$(doctor_value "paths.${name,,}")" "${!var}" \
            || die "doctor reports paths.${name,,} '$(doctor_value "paths.${name,,}")', not ${!var}"
    done
    same_path "$(doctor_value paths.error_profile)" "$CAL/stage_errors.precise.json" \
        || die "doctor reports paths.error_profile '$(doctor_value paths.error_profile)', not $CAL/stage_errors.precise.json"
    same_path "$(dirname "$(doctor_value paths.thresholds)")" "$CAL" \
        || die "doctor reports paths.thresholds '$(doctor_value paths.thresholds)', outside $CAL"
    CONFIG_HASH="$(doctor_value config_hash)"
    [[ -n "$CONFIG_HASH" ]] || die "doctor reports no config_hash"
    pv agreement subset --check || die "record the caption review first (agreement subset --record)"
}

prereg_field() {
    query h2_query.py prereg "$PREREG" "$1" || die "cannot read $1 of $PREREG"
}

# What makes the pre-registration unusable for H2 (nothing when it is usable).
prereg_problem() {
    local rule gate value
    rule="$(prereg_field rule)"
    gate="$(prereg_field calibration_gate)"
    [[ -z "$gate" || "$rule" == "$PROVISIONAL" ]] \
        || { echo "rule '$rule' with a calibration gate"; return; }
    [[ -n "$gate" || "$rule" == "$CALIBRATED" ]] \
        || { echo "rule '$rule', not $CALIBRATED: configs/agreement.toml is not the committed one"; return; }
    value="$(prereg_field device)"
    [[ "$value" == cpu ]] || { echo "device '$value', not cpu"; return; }
    value="$(prereg_field profile)"
    [[ "$value" == precise ]] || { echo "profile '$value', not precise"; return; }
    value="$(prereg_field calibration_artifact_sha256)"
    [[ "$value" == "$CALIB_SHA" ]] \
        || { echo "calibration artifact $value, not A8 v2's $CAL/calibration.json ($CALIB_SHA)"; return; }
    value="$(prereg_field exclusion_record_sha256)"
    [[ "$value" == "$A8V2_RECORD_SHA" ]] \
        || { echo "decision record $value, not A8 v2's $A8V2_RECORD_SHA"; return; }
}

# preflight STEP SUBSET CLIP: one cached clip verified; any error in its checks stops.
preflight() {
    begin "$1" "$PHASE"
    [[ -n "$3" ]] || stop "no $2 clip to preflight"
    call 1 -- verify --dataset videophy --level perceived --subset "$2" --clips "$3"
    local run errors
    run="$(run_of)"
    # verify exits 0 on an uncached clip: its rows abstain with the error in checks.jsonl.
    errors="$(query a8v2_query.py run-errors "$run" 2>&1)" \
        || stop "preflight: $3 failed or is not cached for --device cpu:"$'\n'"$errors"
    note "preflight: $2 $3 cached, run $run"
    row 0
}

# pass PREFIX SUBSET: fetch, extract and verify SUBSET; leaves its verify run in VERIFY_RUN.
pass() {
    begin "$1-fetch" "$PHASE"
    call 0 -- agreement fetch --subset "$2"
    row 0
    begin "$1-extract" "$PHASE"
    call 1 -- extract run --dataset videophy --subset "$2"
    row 0
    begin "$1-verify" "$PHASE"
    call 1 -- verify --dataset videophy --level perceived --subset "$2"
    VERIFY_RUN="$(run_of)"
    row 0
}

note_gaps() {
    local lines
    lines="$(query h2_query.py gaps "$1")" || die "cannot count the input gaps of $1"
    note_block "Input gaps ($1)" <<<"$lines"
}

phase_dev() {
    untested || die "a test comparison is already logged: a freeze now is post hoc" \
        "(agreement freeze --force --declare-deviation, by hand)"
    check_doctor
    save DEV_ENV_SHA "$ENV_SHA"
    save DEV_CONFIG_HASH "$CONFIG_HASH"
    save_artifacts
    preflight pre "$DEV_SUBSET" "${PHYSVERIFY_H2_PREFLIGHT_CLIP:-v1-a8208f4dd41e}"
    pass dev "$DEV_SUBSET"
    save DEV_VERIFY_RUN "$VERIFY_RUN"
    begin freeze "$PHASE"
    # An unchanged frozen state keeps the pre-registration (exit 0); a changed one exits 2.
    call 0 -- agreement freeze --verify-run "$DEV_VERIFY_RUN"
    PREREG="$(sed -n '/^preregistration: /{s///p;q}' "$LAST_LOG")"
    [[ -n "$PREREG" && -f "$PREREG" ]] || stop "no pre-registration in $LAST_LOG"
    local problem field
    problem="$(prereg_problem)"
    [[ -z "$problem" ]] || stop "the pre-registration $PREREG is not H2's: $problem"
    save PREREG "$PREREG"
    save PREREG_SHA "$(sha_of "$PREREG")"
    save FROZEN_COMMIT "$HEAD_COMMIT"
    for field in rule rule_label calibration_gate n_dev dev_majority dev_cut dev_tuned_cut \
        calibration_artifact_sha256 exclusion_record_sha256; do
        note "$field: $(prereg_field "$field")"
    done
    note_gaps "$DEV_SUBSET"
    artifacts_unchanged
    row 0
    echo "dev frozen ($PREREG); review $SUMMARY, then run_h2.sh test"
}

phase_test() {
    local spent="H2 test is spent: a test comparison is already logged; its report is in"
    spent+=" <test verify run>/agreement/ and compare never runs twice"
    [[ -z "${TEST_COMPARED:-}" ]] || die "$spent (TEST_COMPARED in $STATE)"
    untested || die "$spent"
    [[ "$(last_exit freeze)" == 0 ]] || die "the dev phase has not completed: run run_h2.sh dev"
    [[ -n "${PREREG:-}" && -f "$PREREG" && "$(sha_of "$PREREG")" == "${PREREG_SHA:-}" ]] \
        || die "the pre-registration ${PREREG:-?} changed since the dev phase"
    [[ "$HEAD_COMMIT" == "${FROZEN_COMMIT:-}" ]] \
        || die "HEAD is $HEAD_COMMIT, the freeze was at ${FROZEN_COMMIT:-?}"
    [[ "$ENV_SHA" == "${DEV_ENV_SHA:-}" ]] \
        || die "the PHYSVERIFY__* environment differs from the dev phase's (paths or lock moved)"
    artifacts_unchanged
    local problem gate
    problem="$(prereg_problem)"
    [[ -z "$problem" ]] || die "the pre-registration $PREREG is not H2's: $problem"
    gate="$(prereg_field calibration_gate)"
    if [[ -n "$gate" && "${PHYSVERIFY_H2_ACCEPT_NOT_EVALUABLE:-}" != 1 ]]; then
        die "the calibrated stratum was not usable at the freeze ($gate): H2 is not evaluable" \
            "with the calibrated rule (ADR 0015 1.3) whether or not test runs, and redoing A8 or" \
            "the freeze to avoid that is post hoc. Set PHYSVERIFY_H2_ACCEPT_NOT_EVALUABLE=1 to" \
            "spend test on the descriptive result."
    fi
    check_doctor
    [[ "$CONFIG_HASH" == "${DEV_CONFIG_HASH:-}" ]] \
        || die "doctor's config_hash $CONFIG_HASH differs from the dev phase's ${DEV_CONFIG_HASH:-?}"
    local clip="${PHYSVERIFY_H2_PREFLIGHT_TEST_CLIP:-}"
    [[ -n "$clip" ]] || clip="$(query h2_query.py ready "$TEST_SUBSET")" \
        || die "no $TEST_SUBSET clip to preflight"
    preflight test-pre "$TEST_SUBSET" "$clip"
    pass test "$TEST_SUBSET"
    save TEST_VERIFY_RUN "$VERIFY_RUN"
    begin compare "$PHASE"
    call 0 -- agreement compare --verify-run "$TEST_VERIFY_RUN" --subset "$TEST_SUBSET" \
        --rule "$CALIBRATED"
    save TEST_COMPARED 1
    mkdir -p "$TEST_VERIFY_RUN/agreement"
    cp "$PREREG" "$TEST_VERIFY_RUN/agreement/preregistration.json"
    ln -sfn "$(basename "$TEST_VERIFY_RUN")" "$(dirname "$TEST_VERIFY_RUN")/$H2_LINK"
    note "$(sed -n '/^H2: /{p;q}' "$LAST_LOG")"
    note "report: $TEST_VERIFY_RUN/agreement/summary.md"
    note_gaps "$TEST_SUBSET"
    artifacts_unchanged
    row 0
    echo "H2 run: $(dirname "$TEST_VERIFY_RUN")/$H2_LINK (summary: agreement/summary.md)"
}

require_a8v2
"phase_$PHASE"
