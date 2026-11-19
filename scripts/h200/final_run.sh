#!/usr/bin/env bash
# The v3 final test on the H200 in one command, in a clean checkout at the frozen commit:
#
#   bash scripts/h200/final_run.sh [FROM [TO]]     stages FROM..TO (default 0..10); FROM resumes
#
#    0 preflight    checkout, foreign config, pinned render plans, synth code unchanged since the
#                   render, sealed versions outside calibration.development_sets, no ledger line on
#                   the sets this run looks at
#    1 calibrate    run_perception_iteration.sh 1 3, then 4 5 (iteration $ITERATION_ID, calib only)
#    2 freeze       run_freeze.sh 1 7 on that iteration's directory with PHYSVERIFY_FREEZE_SKIP_V1=1
#                   -> $F/freeze.json
#    3 preflight-active   final-table preflight: no weight-0 check can make a clip abstain
#    4 extract      extract run on every Kubric version and real set below (exempt: no verdict)
#    5 look         one calibrate evaluate per set, with PHYSVERIFY_UNSEAL of that set
#    6 attribution  diagnose attribution on the counted Kubric set (same frozen state)
#    7 agreement    run_human_agreement.sh prefetch, dev (agreement freeze on h2_dev),
#                   prefetch-test, then test (the one agreement compare on h2_test) under
#                   $F/freeze.json, in $F/h2 with paths.data_dir $F/h2-data, this run's calibration,
#                   cache, interpreter and threads, and none of its PHYSVERIFY__* (the runner sets
#                   its own); its own freeze:<sha> ledger line. A resume skips prefetch and dev once
#                   their freeze completed, and the test once it compared (it never compares twice)
#    8 table        selective-risk curve, then final-table build -> $F/table/final_table.{md,json}
#    9 vlm          baselines score: AutoEval on the counted Kubric set, then on each real set
#                   (exempt like extract: it fills the score cache $F/vlm/scores, prints no score)
#   10 compare      baselines compare on the counted Kubric set, then on the real sets (two more
#                   looks under this run's freeze, no --set), then final-table comparison ->
#                   $F/table/comparison_table.{md,json}; each step is saved, a resume skips it
#
# Kubric sets (dataset kubric_fresh, subset eval; a version other than the configured one goes as
# --set, which the sealing guard keeps sealed): v3-test is counted; v3-test-r3b and v2-dev-b5 are
# sealed too, read the same way and reported as supplementary. No version of either list may be a
# declared development set (its look would be exempt): the preflight stops on one. Real sets:
# physics_iq_views and objectron_static, through --real-dataset on the calib clips.
#
# Required:
#   PHYSVERIFY_FINAL_EXPECT_COMMIT   prefix (>= 7 chars) of the frozen commit, HEAD of this checkout
#   PHYSVERIFY_FINAL_EARLY_EXTRACTION_COMMIT   run_freeze.sh's early-extraction commit (stage 2)
#   PHYSVERIFY_FINAL_CACHE_DIR       paths.cache_dir (the shared cache)
# Optional (defaults in brackets):
#   PV_DEVICE                        cpu or cuda [cuda]; stages 1, 2, 4, 7 and 9 may need a GPU
#   PHYSVERIFY_FINAL_DIR             $F [$H200_ROOT/final-v3]
#   PHYSVERIFY_FINAL_DATA            paths.data_dir [$H200_ROOT/kubric-data]
#   PHYSVERIFY_FINAL_PYTHON          interpreter [/home/mmonterot/dev/venv/bin/python]
#   PHYSVERIFY_FINAL_THREADS         runtime.threads [16]
#   PHYSVERIFY_FINAL_AUTOEVAL_ROOT   the AutoEval install (venvs/videophy2) stage 9 scores with
#                                    [$H200_ROOT]
#   PHYSVERIFY_FINAL_DRYRUN          1 for a rehearsal on development data only [0]: stages 1-2 are
#                                    skipped, stage 7 reads PHYSVERIFY_FINAL_DRY_ITEMS; $F, the
#                                    ledger (PHYSVERIFY__SEALED__LEDGER, required) and every output
#                                    lie under $H200_ROOT/scratch; the calibration is
#                                    PHYSVERIFY_FINAL_DRY_CALIBRATION (a fitted dir), every Kubric
#                                    set is PHYSVERIFY_FINAL_DRY_KUBRIC [v2-dev-b3], the real sets
#                                    are physics_iq and pisa_real, and the agreement items are
#                                    PHYSVERIFY_FINAL_DRY_ITEMS (an h2_dev items.csv); the
#                                    attribution stage reads only PHYSVERIFY_FINAL_DRY_CLIPS
#                                    (space-separated ids) when set, and stage 9 scores only the
#                                    PHYSVERIFY_FINAL_DRY_VLM_LIMIT clips of least sha256(clip_id)
#                                    of each set when set
# Stages 1 and 2 take their interpreter and threads from the launcher's environment, not from
# PHYSVERIFY_FINAL_PYTHON and _THREADS: PHYSVERIFY_ITERATION_{PYTHON,THREADS} and
# PHYSVERIFY_FREEZE_{PYTHON,THREADS} (their runners' defaults otherwise). Stage 7 passes the
# launcher's other PHYSVERIFY_AGREEMENT_* variables through (ACCEPT_NOT_EVALUABLE,
# PREFLIGHT_TEST_CLIP; run_human_agreement.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steps.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_guards.sh"
cd "$REPO_ROOT"

FROM="${1:-0}"
TO="${2:-10}"
STAGE='^([0-9]|10)$'
[[ "$FROM" =~ $STAGE && "$TO" =~ $STAGE && "$FROM" -le "$TO" ]] \
    || die "usage: final_run.sh [FROM [TO]], 0 <= FROM <= TO <= 10"

# The sealed render (code 58a116d): plan.json sha256 per version, pinned before any look.
RENDER_COMMIT=58a116d
declare -A PLAN_SHA=(
    [v3-test]=b3189682e3857d2e821235e7de140b373195b6cf683dbb90a6c134bbe14700f9
    [v3-test-r3b]=26219d815fd3f2f2d80372b66f2b9ece0130edf7ca0f3070038c27ed08451cda
    [v2-dev-b5]=e3d26aeba9286a52d4e6696d3c156043be8d768be03b59bf22b7903e561c31b2
)
SYNTH_PATHS=(src/physverify/synth src/physverify/kubric_worker configs/datasets/kubric_synthetic.toml)
COUNTED=v3-test
SUPPLEMENTARY=(v3-test-r3b v2-dev-b5)
REAL=(physics_iq_views objectron_static)
# The sealed versions of a real run; a rehearsal reads development sets but still checks these.
SEALED_KUBRIC=("$COUNTED" "${SUPPLEMENTARY[@]}")
# The iteration stage calibrate runs and stage freeze adopts.
ITERATION_ID=it6-v3final

source_h200_env
DRYRUN="${PHYSVERIFY_FINAL_DRYRUN:-0}"
[[ "$DRYRUN" == 0 || "$DRYRUN" == 1 ]] || die "PHYSVERIFY_FINAL_DRYRUN must be 0 or 1"
SCRATCH="$H200_ROOT/scratch"
F="${PHYSVERIFY_FINAL_DIR:-$H200_ROOT/final-v3}"
DATA="${PHYSVERIFY_FINAL_DATA:-$H200_ROOT/kubric-data}"
CACHE="${PHYSVERIFY_FINAL_CACHE_DIR:?set PHYSVERIFY_FINAL_CACHE_DIR to the shared cache}"
PY="${PHYSVERIFY_FINAL_PYTHON:-/home/mmonterot/dev/venv/bin/python}"
THREADS="${PHYSVERIFY_FINAL_THREADS:-16}"
CAL="$F/calibration"
VLM="$F/vlm"
AUTOEVAL_ROOT="${PHYSVERIFY_FINAL_AUTOEVAL_ROOT:-$H200_ROOT}"
VLM_LIMIT=()
PV_RUNNER=py
OWN=" PATHS__DATA_DIR PATHS__CACHE_DIR PATHS__WEIGHTS_DIR PATHS__VENDOR_DIR PATHS__HF_HOME
    PATHS__CALIBRATION_DIR PATHS__RUNS_DIR RUNTIME__HEAVY_LOCK "
if [[ "$DRYRUN" == 1 ]]; then
    CAL="${PHYSVERIFY_FINAL_DRY_CALIBRATION:?a dry run needs PHYSVERIFY_FINAL_DRY_CALIBRATION}"
    [[ -f "$CAL/abstention.json" ]] || die "$CAL has no abstention.json (selective-risk fit)"
    [[ -n "${PHYSVERIFY__SEALED__LEDGER:-}" ]] || die "a dry run needs its own ledger under $SCRATCH"
    for dir in "$F" "$PHYSVERIFY__SEALED__LEDGER"; do
        under "$dir" "$SCRATCH" || die "a dry run writes only under $SCRATCH, not $dir"
    done
    COUNTED="${PHYSVERIFY_FINAL_DRY_KUBRIC:-v2-dev-b3}"
    [[ "$COUNTED" != v3-test* ]] || die "a dry run reads development data only, not $COUNTED"
    SUPPLEMENTARY=()
    REAL=(physics_iq pisa_real)
    if [[ -n "${PHYSVERIFY_FINAL_DRY_VLM_LIMIT:-}" ]]; then
        [[ "$PHYSVERIFY_FINAL_DRY_VLM_LIMIT" =~ ^[1-9][0-9]*$ ]] \
            || die "PHYSVERIFY_FINAL_DRY_VLM_LIMIT must be a positive integer"
        VLM_LIMIT=(--limit "$PHYSVERIFY_FINAL_DRY_VLM_LIMIT")
    fi
    refuse_foreign_config final_run.sh "$OWN" '^PHYSVERIFY__SEALED__LEDGER$'
else
    ! near "$F" "$SCRATCH" || die "$F is under $SCRATCH: only a dry run writes there"
    refuse_foreign_config final_run.sh "$OWN"
fi
# The real sets as a look's PHYSVERIFY_UNSEAL list and as a command's --real-dataset options.
REAL_UNSEAL="$(IFS=,; echo "${REAL[*]}")"
REAL_OPTIONS=()
for name in "${REAL[@]}"; do REAL_OPTIONS+=(--real-dataset "$name"); done
for name in PHYSVERIFY_UNSEAL PHYSVERIFY_UNSEAL_REPEAT; do
    [[ -z "${!name+set}" ]] || die "$name is set: final_run.sh sets it for each look only"
done
refuse_local_config final_run.sh
require_checkout PHYSVERIFY_FINAL_EXPECT_COMMIT
use_device PV_DEVICE
ITERATION_DIR="$(iteration_dir "$ITERATION_ID" "$PV_DEVICE")"

mkdir -p "$F/run" "$F/table"
STATE="$F/run/state.env"
STEPS="$F/run/steps.tsv"
NOTES="$F/run/notes.md"
SUMMARY="$F/run/summary.md"
LOGS="$F/run/logs"
SUMMARY_TITLE="final_run.sh"
SUMMARY_INFO="commit $HEAD_COMMIT, device $PV_DEVICE, dry run $DRYRUN"
mkdir -p "$LOGS"
touch "$NOTES" "$STEPS"
[[ ! -f "$STATE" ]] || source "$STATE"
[[ "${RUN_COMMIT:-$HEAD_COMMIT}" == "$HEAD_COMMIT" ]] || die "$STATE is of commit $RUN_COMMIT"
# Heavy-stage cache keys hold the device: a resume on another device finds no input, silently.
[[ "${RUN_DEVICE:-$PV_DEVICE}" == "$PV_DEVICE" ]] || die "$STATE is of device $RUN_DEVICE"
declare -A STEP_SAVES=([5]="KUBRIC_EVAL REAL_EVAL" [6]="ATTRIBUTION" [7]="AGREEMENT_ITEMS")

# save_file KEY PATH: save an output this stage must have produced.
save_file() {
    [[ -n "$2" && -f "$2" ]] || stop "no $1 output (got '$2')"
    save "$1" "$2"
}

# save_evaluation KEY PATH: save_file, then report (never stop: the look is spent) any active
# check of PATH left without input; stage extract already refused a failed clip before the look.
save_evaluation() {
    save_file "$1" "$2"
    call 1 -- final-table inputs --evaluation "$2" --preflight "$F/table/preflight.json"
}
require_completed 0 "$FROM"

export PHYSVERIFY__PATHS__DATA_DIR="$DATA"
export PHYSVERIFY__PATHS__CACHE_DIR="$CACHE"
export PHYSVERIFY__PATHS__WEIGHTS_DIR="$H200_ROOT/Repo/data/weights"
export PHYSVERIFY__PATHS__VENDOR_DIR="$H200_ROOT/Repo/data/vendor"
export PHYSVERIFY__PATHS__HF_HOME="$H200_ROOT/.cache/huggingface"
export PHYSVERIFY__PATHS__CALIBRATION_DIR="$CAL"
export PHYSVERIFY__PATHS__RUNS_DIR="$F/runs"
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$F/heavy.lock"

# kubric VERSION ARGS: a pv call on kubric_fresh at VERSION (configured or --set).
kubric() {
    local version="$1"
    shift
    PV_EXTRA=()
    [[ "$version" == "$(configured_version)" ]] || PV_EXTRA=(--set "kubric_fresh.version=$version")
    call 0 -- "$@"
    PV_EXTRA=()
}

configured_version() {
    "$PY" -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["kubric_fresh"]["version"])' \
        configs/datasets/kubric_fresh.toml
}

# look SPECS CHECK ARGS: ARGS with PHYSVERIFY_UNSEAL=SPECS (comma separated); with CHECK=1 refused
# when a ledger line already covers them (look-state reads the configured version, so the
# supplementary versions, set by --set, are only recorded, not refused).
look() {
    local specs="$1" check="$2"
    shift 2
    if [[ "$DRYRUN" == 0 && "$check" == 1 ]]; then
        fq look-state ${specs//,/ } >/dev/null || stop "one look per set: $specs"
    fi
    PHYSVERIFY_UNSEAL="$specs" "$@"
}

stage_preflight() {
    local version dirty
    save RUN_COMMIT "$HEAD_COMMIT"
    save RUN_DEVICE "$PV_DEVICE"
    if [[ "$DRYRUN" == 0 ]]; then
        for version in "$COUNTED" "${SUPPLEMENTARY[@]}"; do
            [[ -n "${PLAN_SHA[$version]}" ]] || stop "no pinned plan.json sha256 for $version"
            [[ "$(sha_of "$DATA/synthetic/$version/plan.json")" == "${PLAN_SHA[$version]}" ]] \
                || stop "plan.json of $version changed since the render"
        done
        dirty="$(git diff --stat "$RENDER_COMMIT" HEAD -- "${SYNTH_PATHS[@]}")"
        [[ -z "$dirty" ]] || stop "scene generation changed since the render: $dirty"
        fq ledger >/dev/null || stop "the sealed ledger is unreadable (fail closed)"
        fq look-state kubric_fresh "${REAL[@]}" >/dev/null || stop "a set of this run was looked at"
    fi
    env PYTHONPATH="$REPO_ROOT/src" "$PY" - "${SEALED_KUBRIC[@]}" <<'PY' \
        || stop "a sealed Kubric version (${SEALED_KUBRIC[*]}) is a declared development set"
import sys
from pathlib import Path
from physverify.calibration.settings import calibration_settings
from physverify.config.loader import load_settings
declared = set(calibration_settings(load_settings(Path("configs"), environ={})).development_sets)
raise SystemExit(any(f"kubric_fresh@{version}" in declared for version in sys.argv[1:]))
PY
    note "preflight passed (plans, synth diff, ledger, development sets)"
}

# iterate FROM TO: steps FROM..TO of this run's calibration iteration.
iterate() {
    PHYSVERIFY_ITERATION_ID="$ITERATION_ID" PHYSVERIFY_ITERATION_DIR="$ITERATION_DIR" \
        PHYSVERIFY_ITERATION_EXPECT_COMMIT="$HEAD_COMMIT" PHYSVERIFY_ITERATION_CACHE_DIR="$CACHE" \
        PHYSVERIFY_ITERATION_DEVICE="$PV_DEVICE" \
        run_logged 0 -- bash scripts/h200/run_perception_iteration.sh "$1" "$2"
}

# Two ranges: the iteration refuses a cuda range that crosses from its GPU steps to its cache-only
# ones (_common.sh use_step_device).
stage_calibrate() {
    iterate 1 3
    iterate 4 5
}

stage_freeze() {
    PHYSVERIFY_FREEZE_EXPECT_COMMIT="$HEAD_COMMIT" PHYSVERIFY_FREEZE_ITERATION="$ITERATION_ID" \
        PHYSVERIFY_FREEZE_ITERATIONS="$ITERATION_DIR" PHYSVERIFY_FREEZE_SKIP_V1=1 \
        PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT="${PHYSVERIFY_FINAL_EARLY_EXTRACTION_COMMIT:?}" \
        PHYSVERIFY_FREEZE_CACHE_DIR="$CACHE" PHYSVERIFY_FREEZE_DIR="$F" \
        run_logged 0 -- bash scripts/h200/run_freeze.sh 1 7
}

stage_active() {
    call 0 -- final-table preflight --artifact "$CAL/calibration.json" --out "$F/table/preflight.json"
}

stage_extract() {
    local version name
    for version in "$COUNTED" "${SUPPLEMENTARY[@]}"; do
        kubric "$version" extract run --dataset kubric_fresh --subset eval
    done
    for name in "${REAL[@]}"; do
        call 0 -- extract run --dataset "$name"
    done
}

stage_look() {
    local version
    look kubric_fresh 1 kubric "$COUNTED" calibrate evaluate --artifact "$CAL/calibration.json" \
        --dataset kubric_fresh --subset eval --no-real
    save_evaluation KUBRIC_EVAL "$(evaluation_of "$LAST_LOG")"
    for version in "${SUPPLEMENTARY[@]}"; do
        look kubric_fresh 0 kubric "$version" calibrate evaluate \
            --artifact "$CAL/calibration.json" --dataset kubric_fresh --subset eval --no-real
        save_evaluation "SUPPLEMENTARY_${version//-/_}" "$(evaluation_of "$LAST_LOG")"
    done
    look "$REAL_UNSEAL" 1 call 0 -- calibrate evaluate --artifact "$CAL/calibration.json" \
        --dataset kubric_synthetic --subset calib "${REAL_OPTIONS[@]}"
    save_evaluation REAL_EVAL "$(evaluation_of "$LAST_LOG")"
}

stage_attribution() {
    local id clips=()
    if [[ "$DRYRUN" == 1 ]]; then
        for id in ${PHYSVERIFY_FINAL_DRY_CLIPS:-}; do clips+=(--clips "$id"); done
    fi
    PHYSVERIFY_UNSEAL=kubric_fresh kubric "$COUNTED" diagnose attribution \
        --dataset kubric_fresh --subset eval "${clips[@]}"
    save_file ATTRIBUTION "$(find "$(run_of)" -name attribution.json | head -1)"
}

# agreement PHASE: run_human_agreement.sh PHASE under this run's freeze. It refuses any
# PHYSVERIFY__* it does not export itself, so this run's are unset; its data directory is new,
# outside $F/h2 (the runner's default holds the v1 study's comparison).
agreement() {
    local name unset=()
    for name in $(compgen -e -X '!PHYSVERIFY__*'); do unset+=(-u "$name"); done
    run_logged 0 -- env "${unset[@]}" PHYSVERIFY_AGREEMENT_FREEZE="$F/freeze.json" \
        PHYSVERIFY_AGREEMENT_DIR="$F/h2" PHYSVERIFY_AGREEMENT_DATA="$F/h2-data" \
        PHYSVERIFY_AGREEMENT_CALIBRATION_DIR="$CAL" PHYSVERIFY_AGREEMENT_CACHE_DIR="$CACHE" \
        PHYSVERIFY_AGREEMENT_EXPECT_COMMIT="$HEAD_COMMIT" PHYSVERIFY_AGREEMENT_DEVICE="$PV_DEVICE" \
        PHYSVERIFY_AGREEMENT_PYTHON="$PY" PHYSVERIFY_AGREEMENT_THREADS="$THREADS" \
        bash scripts/h200/run_human_agreement.sh "$1"
}

stage_agreement() {
    if [[ "$DRYRUN" == 1 ]]; then
        save_file AGREEMENT_ITEMS "${PHYSVERIFY_FINAL_DRY_ITEMS:?a dry run needs PHYSVERIFY_FINAL_DRY_ITEMS}"
        return
    fi
    local study="$F/h2"
    if [[ "$(last_exit freeze "$study/steps.tsv")" != 0 ]]; then
        agreement prefetch
        agreement dev
    fi
    if [[ -z "$(state_value "$study/state.env" TEST_COMPARED)" ]]; then
        agreement prefetch-test
        agreement test
    fi
    save_file AGREEMENT_ITEMS "$(state_value "$study/state.env" TEST_VERIFY_RUN)/agreement/items.csv"
}

# curve: the held-out selective-risk curve on the counted set and the real sets, a second look
# under this run's freeze (the first was stage look): the sets go as options, never as --set,
# which would change the configuration and with it the frozen state.
curve() {
    look "kubric_fresh,$REAL_UNSEAL" 0 kubric "$COUNTED" selective-risk curve \
        --dataset kubric_fresh --subset eval "${REAL_OPTIONS[@]}"
}

stage_table() {
    local version curve key extra=()
    curve
    curve="$(find "$(run_of)" -name abstention_report.json | head -1)"
    for version in "${SUPPLEMENTARY[@]}"; do
        key="SUPPLEMENTARY_${version//-/_}"
        if [[ "$version" == v3-test-r3b ]]; then extra+=(--r3b "${!key}")
        else extra+=(--replication "$version=${!key}"); fi
    done
    if [[ "$DRYRUN" == 1 ]]; then extra+=(--dry-run); else extra+=(--freeze "$F/freeze.json"); fi
    call 0 -- final-table build --out "$F/table" --kubric "$KUBRIC_EVAL" --real "$REAL_EVAL" \
        --curve "$curve" --agreement-items "$AGREEMENT_ITEMS" --preflight "$F/table/preflight.json" \
        --attribution "$ATTRIBUTION" "${extra[@]}"
}

# vlm: AutoEval's score of every clip stage look read (the same clips, through the same options),
# cached per clip under $VLM/scores; no score is printed and nothing reads one before stage 10.
stage_vlm() {
    local name score=(baselines score --judge autoeval)
    mkdir -p "$VLM/scores"
    kubric "$COUNTED" "${score[@]}" --dataset kubric_fresh --subset eval "${VLM_LIMIT[@]}" \
        --scores "$VLM/scores" --autoeval-root "$AUTOEVAL_ROOT"
    for name in "${REAL[@]}"; do
        call 0 -- "${score[@]}" --dataset "$name" "${VLM_LIMIT[@]}" \
            --scores "$VLM/scores" --autoeval-root "$AUTOEVAL_ROOT"
    done
}

# once KEY FUNCTION: FUNCTION, which saves KEY, unless an earlier run of this stage saved KEY (a
# resume never repeats a comparison's look).
once() {
    if [[ -n "${!1:-}" ]]; then
        note "stage $CUR_NAME: $1 saved earlier, not repeated (${!1})"
        return
    fi
    "$2"
}

# compare_kubric, compare_real: the comparisons of the counted Kubric set and of the real sets, two
# more looks under this run's freeze (the same artifact, configuration and options as stage look).
compare_kubric() {
    look kubric_fresh 0 kubric "$COUNTED" baselines compare --part kubric \
        --artifact "$CAL/calibration.json" --dataset kubric_fresh --subset eval \
        --scores "$VLM/scores" --out "$VLM"
    save_file VLM_KUBRIC "$(printed comparison "$LAST_LOG")"
}

compare_real() {
    look "$REAL_UNSEAL" 0 call 0 -- baselines compare --part real \
        --artifact "$CAL/calibration.json" "${REAL_OPTIONS[@]}" --scores "$VLM/scores" --out "$VLM"
    save_file VLM_REAL "$(printed comparison "$LAST_LOG")"
}

comparison_table() {
    call 0 -- final-table comparison --out "$F/table" --kubric "$VLM_KUBRIC" --real "$VLM_REAL" \
        --agreement-items "$AGREEMENT_ITEMS" --final-table "$F/table/final_table.json"
    save_file COMPARISON_TABLE "$(printed "comparison table" "$LAST_LOG")"
}

stage_compare() {
    once VLM_KUBRIC compare_kubric
    once VLM_REAL compare_real
    once COMPARISON_TABLE comparison_table
}

STAGES=(preflight calibrate freeze active extract look attribution agreement table vlm compare)
for ((n = FROM; n <= TO; n++)); do
    name="${STAGES[$n]}"
    begin "$n" "$name"
    if [[ "$DRYRUN" == 1 && " calibrate freeze " == *" $name "* ]]; then
        note "dry run: stage $name skipped (calibration $CAL)"
    else
        "stage_$name"
    fi
    row 0
done
