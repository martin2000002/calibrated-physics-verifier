#!/usr/bin/env bash
# A8 v2 on the H200 (ADR 0015, "Corrida CPU"; CUDA as a device sensitivity, see below): the pair-collision migration, the stage
# errors and sigma, the threshold fit and evaluation (with the H1 primary scene view, 1.4), the
# abstention, both R2 detector arms, the exclusion record of 1.1, 1.2 and 1.5, then the Objectron
# size evaluation and the LIFT gravity rows (neither feeds the record, so they come after it).
# It consumes the perception cache the CPU shards left (run_real_cpu_shards.sh,
# run_kubric_cpu_shards.sh) and, by default, never uses a GPU: every call is --profile precise --device cpu,
# the device the cache keys were computed with. With PHYSVERIFY_A8V2_DEVICE=cuda the same chain
# runs on the GPU the launcher left in CUDA_VISIBLE_DEVICES (--device cuda, the cache the GPU
# extraction left) as a device sensitivity: CPU and CUDA magnitudes differ, so it never shares a
# calibration with the CPU chain. Its error profile, calibration artifacts, abstention, exclusion
# record, runs and heavy lock are its own (paths.calibration_dir = $DATA/calibration-cuda,
# paths.runs_dir = $DATA/runs-cuda, runtime.heavy_lock in its output directory); the
# pair-collision migration is shared, so it only checks that the CPU chain completed it.
#
#   bash scripts/h200/run_a8v2.sh [FROM [TO]]   steps FROM..TO (default 0..11); a TO below 11
#                                               stops there with exit 0, `FROM=TO+1` resumes
#
#   startup  no other instance on $PHYSVERIFY_A8V2_DIR (run.lock); no running extract, dataset or
#            diagnose job on a dataset steps FROM..TO touch (STEP_DATASETS; one whose --dataset
#            cannot be read always blocks), checked again before every step; extract and diagnose
#            jobs of the other device are ignored (the cache keys hold the device), a job whose
#            device cannot be read blocks, dataset jobs block on either device; HEAD is
#            PHYSVERIFY_A8V2_EXPECT_COMMIT with no uncommitted change under src, configs or
#            scripts; paths.cache_dir is PHYSVERIFY_A8V2_CACHE_DIR; steps 0..FROM-1 last recorded
#            with exit 0 in steps.tsv and their STEP_SAVES in state.env; when the range touches
#            pisa_real and FROM <= 7, the PISA selection is within what the shards extracted
#   pre      (FROM <= 8) one cached clip per dataset of the range verified (any error in its
#            checks.jsonl stops), so a wrong cache stops here and not after days of recomputation;
#            before step 2 has begun, one timed `diagnose stages` calib clip not timed before (cost
#            of steps 2, 4)
#    0 migrate            migrate_velocity.sh with pair-collision (needs docker); skipped when its
#                         manifest is complete and its audit.json exists, so a rerun never purges
#    1 extract-synthetic  re-extracts kubric_synthetic/full (only the re-rendered clips are new)
#    2 stages             diagnose stages --write-profile (the error profile); with
#                         PHYSVERIFY_A8V2_DIAG_SHARDS first that many parallel `diagnose stages`
#                         shards of the full selection fill the heavy cache, which does not depend
#                         on the error profile, and the single profile-writing call reads it
#    3 relift             extract run again on kubric_synthetic, physics_iq and pisa_real (LIFT
#                         reads the new profile)
#    4 stage-sigma        calibrate stage-sigma
#    5 fit                calibrate fit --level both
#    6 evaluate           calibrate evaluate --real; stops on a skipped real dataset, a real set
#                         1.5 does not allow or a missing H1 primary view
#    7 abstain            abstain fit, abstain curve
#    8 r2-arms            the R2 clips verified with the base and the tiny detector (rule 1.2)
#    9 decide             exclusions decide, then exclusions show (the record's sha256)
#   10 size-eval          size eval (Objectron)
#   11 gravity            the calib LIFT gravity-angle rows of step 2, copied to the summary
#
# Steps 2 and 4 compute the diagnosis ladder variants (V1: TRACK on the true DETECT, a heavy
# stage the shards never ran) on the chain's device under the lazy heavy lock; the preflight does not cover
# them, it only times one clip of them and writes the estimate to the summary.
# Everything lands in $PHYSVERIFY_A8V2_DIR (default $H200_ROOT/a8v2, cuda: a8v2-cuda): logs/,
# steps.tsv, summary.md and state.env (run ids later steps need, so FROM resumes). Exit codes:
# 0 ok, 2 a check failed, otherwise the failing command's own (1 failed clips where not tolerated,
# 75 lock).
#
# Required:
#   PHYSVERIFY_A8V2_EXPECT_COMMIT   prefix (>= 7 chars) of the HEAD this checkout must be at
#   PHYSVERIFY_A8V2_CACHE_DIR       the cache root the shards filled; doctor must report it
# Optional:
#   PHYSVERIFY_A8V2_DEVICE          cpu (default) or cuda
#   PHYSVERIFY_A8V2_PYTHON          interpreter (default /home/mmonterot/dev/venv/bin/python)
#   PHYSVERIFY_A8V2_DATA            paths.data_dir (default $H200_ROOT/kubric-data)
#   PHYSVERIFY_A8V2_DIR             output directory (default $H200_ROOT/a8v2)
#   PHYSVERIFY_A8V2_SHARDS          ids files of the shards (default $H200_ROOT/logs/shards)
#   PHYSVERIFY_A8V2_THREADS         runtime.threads (default 32)
#   PHYSVERIFY_A8V2_VERSION         synthetic dataset version (default v1-precise)
#   PHYSVERIFY_A8V2_PISA_MAX_CLIPS  pisa_real.max_clips (default: the precise profile's, all)
#   PHYSVERIFY_A8V2_PREFLIGHT_KUBRIC / _PHYSICS_IQ / _PISA   preflight clip ids
#   PHYSVERIFY_A8V2_DECIDE_REPLACE  1 to supersede a different decision record
#   PHYSVERIFY_A8V2_DIAG_SHARDS     step 2 warm-up shards (default: none), --no-lock under nice
#   PHYSVERIFY_A8V2_DIAG_THREADS    runtime.threads of each shard (default THREADS / SHARDS)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"

LAST_STEP=11
[[ $# -le 2 ]] || die "usage: run_a8v2.sh [FROM [TO]]"
FROM="${1:-0}"
TO="${2:-$LAST_STEP}"
STEP_NUMBER='^(0|[1-9][0-9]*)$'
[[ "$FROM" =~ $STEP_NUMBER && "$TO" =~ $STEP_NUMBER ]] \
    && [[ "$FROM" -le "$TO" && "$TO" -le "$LAST_STEP" ]] \
    || die "FROM and TO must be step numbers with FROM <= TO <= $LAST_STEP, got '$FROM' '$TO'"
DEVICE="${PHYSVERIFY_A8V2_DEVICE:-cpu}"
case "$DEVICE" in
    cpu) OUT_NAME=a8v2 ;;
    cuda) OUT_NAME=a8v2-cuda ;;
    *) die "PHYSVERIFY_A8V2_DEVICE must be cpu or cuda, got '$DEVICE'" ;;
esac

# The datasets whose extraction cache each step reads or writes: 3 extracts the real sets, 6 and 7
# (abstain curve) read them, 9 reads the R2 clips; 10 (Objectron heads) and 11 (step 2's
# stages.json) read none.
SYNTH=kubric_synthetic
ALL_SETS="$SYNTH physics_iq pisa_real"
STEP_DATASETS=(
    [0]=$SYNTH [1]=$SYNTH [2]=$SYNTH [3]=$ALL_SETS [4]=$SYNTH [5]=$SYNTH [6]=$ALL_SETS
    [7]=$ALL_SETS [8]=$SYNTH [9]=$SYNTH [10]="" [11]=""
)
RANGE_DATASETS=""
for ((n = FROM; n <= TO; n++)); do
    for dataset in ${STEP_DATASETS[$n]}; do
        [[ " $RANGE_DATASETS " == *" $dataset "* ]] \
            || RANGE_DATASETS+="${RANGE_DATASETS:+ }$dataset"
    done
done

in_range() {
    [[ " $RANGE_DATASETS " == *" $1 "* ]]
}

# The state.env keys a step leaves for later ones.
STEP_SAVES=([2]="STAGES_RUN" [6]="EVAL_JSON" [8]="R2_BASE_RUN R2_TINY_RUN" [9]="RECORD_SHA")

OUT="${PHYSVERIFY_A8V2_DIR:-$H200_ROOT/$OUT_NAME}"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || die "another run_a8v2.sh holds $OUT/run.lock"

BUSY_JOBS='physverify(\.cli\.main)?.*(extract run|dataset (build|migrate)|diagnose [a-z_]+)'
KUBRIC_JOBS='dataset (build|migrate)'
DATASET_NAME='^[a-z][a-z0-9_]*$'

# last_option OPTION ARGS: the value of the last `OPTION VALUE` or `OPTION=VALUE` in ARGS.
last_option() {
    grep -oE -- "$1[= ][^ ]*" <<<"$2" | sed -nE "$ s/^$1[= ]//p" || true
}

# The device of a job's command line: the last `--set runtime.device=`, else the last --device;
# empty when neither is readable, since its default comes from configuration the guard cannot see.
job_device() {
    local value
    value="$(last_option '--set[= ]runtime\.device' "$1")"
    [[ -n "$value" ]] || value="$(last_option --device "$1")"
    case "$value" in
        cpu) echo cpu ;;
        cuda | cuda:[0-9]*) echo cuda ;;
    esac
}

# Whether a busy job's command line touches a dataset of the range; one with no readable
# --dataset (or `all`) does (fail safe), and `dataset build|migrate` are kubric_synthetic's on
# either device. An extract or diagnose job of the other device fills other cache keys.
blocks() {
    local args="$1" value values device
    if [[ "$args" =~ $KUBRIC_JOBS ]]; then
        in_range kubric_synthetic
        return
    fi
    device="$(job_device "$args")"
    # The migration (step 0) purges the cache of every device, so then any device blocks.
    ((FROM == 0)) || [[ -z "$device" || "$device" == "$DEVICE" ]] || return 1
    values="$(grep -oE -- '--dataset[= ][^ ]*' <<<"$args" | sed -E 's/^--dataset[= ]//' || true)"
    [[ -n "$values" ]] || return 0
    while IFS= read -r value; do
        [[ "$value" =~ $DATASET_NAME && "$value" != all ]] || return 0
        ! in_range "$value" || return 0
    done <<<"$values"
    return 1
}

# The shards ran with --no-lock, so the heavy lock cannot tell whether they are done.
busy_guard() {
    local pid args busy=""
    while read -r pid args; do
        [[ -n "$pid" && "$pid" != "$$" && "$args" =~ $BUSY_JOBS ]] || continue
        ! blocks "$args" || busy+="$pid $args"$'\n'
    done <<<"$(ps -eo pid=,args=)"
    [[ -z "$busy" ]] \
        || die "jobs on the datasets of steps $FROM-$TO are still running:"$'\n'"$busy"
}
busy_guard

EXPECT="${PHYSVERIFY_A8V2_EXPECT_COMMIT:-}"
[[ ${#EXPECT} -ge 7 ]] || die "set PHYSVERIFY_A8V2_EXPECT_COMMIT to the commit this run uses"
HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die "$REPO_ROOT is not a git checkout"
[[ "$HEAD_COMMIT" == "$EXPECT"* ]] || die "HEAD is $HEAD_COMMIT, expected $EXPECT"
git -C "$REPO_ROOT" diff --quiet HEAD -- src configs scripts \
    || die "uncommitted changes under src, configs or scripts: the run would not be $HEAD_COMMIT"
PY="${PHYSVERIFY_A8V2_PYTHON:-/home/mmonterot/dev/venv/bin/python}"
[[ -x "$PY" ]] || die "no interpreter at $PY (PHYSVERIFY_A8V2_PYTHON)"
EXPECT_CACHE="${PHYSVERIFY_A8V2_CACHE_DIR:-}"
[[ -n "$EXPECT_CACHE" ]] || die "set PHYSVERIFY_A8V2_CACHE_DIR to the cache root of the shards"

source_h200_env
export PHYSVERIFY__PATHS__DATA_DIR="${PHYSVERIFY_A8V2_DATA:-$H200_ROOT/kubric-data}"
export PHYSVERIFY__PATHS__RAW_DIR="$H200_ROOT/a8/raw"
export PHYSVERIFY__PATHS__WEIGHTS_DIR="$H200_ROOT/Repo/data/weights"
export PHYSVERIFY__PATHS__VENDOR_DIR="$H200_ROOT/Repo/data/vendor"
export PHYSVERIFY__PATHS__HF_HOME="$H200_ROOT/.cache/huggingface"
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$H200_ROOT/kubric.lock"
if [[ "$DEVICE" == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=""
else
    [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] \
        || die "CUDA_VISIBLE_DEVICES must name the one GPU the launcher chose (got '${CUDA_VISIBLE_DEVICES-unset}')"
    # Every artifact outside $OUT derives from these two (doctor confirms it below).
    export PHYSVERIFY__PATHS__CALIBRATION_DIR="$PHYSVERIFY__PATHS__DATA_DIR/calibration-cuda"
    export PHYSVERIFY__PATHS__RUNS_DIR="$PHYSVERIFY__PATHS__DATA_DIR/runs-cuda"
    unset PHYSVERIFY__PATHS__ERROR_PROFILE PHYSVERIFY__PATHS__THRESHOLDS
    export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$OUT/heavy.lock"
fi
unset PHYSVERIFY_LOCK_HELD
PV_DEVICE="$DEVICE"
THREADS="${PHYSVERIFY_A8V2_THREADS:-32}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "PHYSVERIFY_A8V2_THREADS must be a positive integer"
PISA_SET=()
if [[ -n "${PHYSVERIFY_A8V2_PISA_MAX_CLIPS:-}" ]]; then
    [[ "$PHYSVERIFY_A8V2_PISA_MAX_CLIPS" =~ ^[1-9][0-9]*$ ]] \
        || die "PHYSVERIFY_A8V2_PISA_MAX_CLIPS must be a positive integer"
    PISA_SET=(--set "pisa_real.max_clips=$PHYSVERIFY_A8V2_PISA_MAX_CLIPS")
fi
KS=(--dataset kubric_synthetic --subset full)
DIAG_SHARDS="${PHYSVERIFY_A8V2_DIAG_SHARDS:-}"
DIAG_THREADS=""
if [[ -n "$DIAG_SHARDS" ]]; then
    [[ "$DIAG_SHARDS" =~ ^[1-9][0-9]*$ ]] \
        || die "PHYSVERIFY_A8V2_DIAG_SHARDS must be a positive integer, got '$DIAG_SHARDS'"
    per_shard=$((THREADS / DIAG_SHARDS))
    DIAG_THREADS="${PHYSVERIFY_A8V2_DIAG_THREADS:-$((per_shard > 0 ? per_shard : 1))}"
    [[ "$DIAG_THREADS" =~ ^[1-9][0-9]*$ ]] \
        || die "PHYSVERIFY_A8V2_DIAG_THREADS must be a positive integer, got '$DIAG_THREADS'"
fi
# A prefix of every physverify call (the step 2 shards run under nice).
PV_NICE=()

pv() {
    "${PV_NICE[@]}" env PYTHONPATH="$REPO_ROOT/src" "$PY" -c "from physverify.cli.main import main; main()" \
        --profile precise --device "$PV_DEVICE" --set "runtime.threads=$THREADS" \
        "${PISA_SET[@]}" "$@"
}

query() {
    env PYTHONPATH="$REPO_ROOT/src" "$PY" "$REPO_ROOT/scripts/h200/a8v2_query.py" \
        "${PISA_SET[@]}" "$@"
}

doctor_value() {
    sed -n "s/^$1: //p" <<<"$DOCTOR" | sed 's/ (.*)$//'
}

same_path() {
    [[ -n "$1" && "$(realpath -m "$1")" == "$(realpath -m "$2")" ]]
}

# The cuda chain writes nothing where the CPU chain does: its device, calibration directory (error
# profile, artifact, stage sigma, abstention, exclusion record) and runs directory are its own.
cuda_paths() {
    local name dir calib="$PHYSVERIFY__PATHS__CALIBRATION_DIR"
    [[ "$(doctor_value device)" == cuda* ]] \
        || die "doctor reports device '$(doctor_value device)', not cuda"
    for name in CALIBRATION_DIR RUNS_DIR; do
        dir="PHYSVERIFY__PATHS__$name"
        same_path "$(doctor_value "paths.${name,,}")" "${!dir}" \
            || die "doctor reports paths.${name,,} '$(doctor_value "paths.${name,,}")', not ${!dir}"
    done
    for name in error_profile thresholds; do
        same_path "$(dirname "$(doctor_value "paths.$name")")" "$calib" \
            || die "doctor reports paths.$name '$(doctor_value "paths.$name")', outside $calib"
    done
}

DOCTOR="$(pv doctor)" || die "physverify doctor failed"
CACHE_DIR="$(doctor_value paths.cache_dir)"
[[ -n "$CACHE_DIR" && "$(realpath -m "$CACHE_DIR")" == "$(realpath -m "$EXPECT_CACHE")" ]] \
    || die "doctor reports paths.cache_dir '$CACHE_DIR', not the shards' $EXPECT_CACHE"
# The profile of 1.19: the precise one, never a recalibrated copy.
ERROR_PROFILE="$(doctor_value paths.error_profile)"
[[ "$(basename "$ERROR_PROFILE")" == stage_errors.precise.json ]] \
    || die "doctor reports paths.error_profile '$ERROR_PROFILE', not stage_errors.precise.json"
[[ "$DEVICE" == cpu ]] || cuda_paths
SYNTH_DIR="$(doctor_value paths.synthetic_dir)"
VERSION="${PHYSVERIFY_A8V2_VERSION:-v1-precise}"
MIGRATION=pair-collision
MANIFEST="$SYNTH_DIR/$VERSION/migrations/$MIGRATION.json"
AUDIT="$(doctor_value paths.data_dir)/runs/migrate-$MIGRATION-$VERSION/audit.json"
SHARDS="${PHYSVERIFY_A8V2_SHARDS:-$H200_ROOT/logs/shards}"

LOGS="$OUT/logs"
STATE="$OUT/state.env"
STEPS="$OUT/steps.tsv"
NOTES="$OUT/notes.md"
SUMMARY="$OUT/summary.md"
mkdir -p "$LOGS"
touch "$STEPS" "$NOTES"
if [[ -f "$STATE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE"
fi

# --- state and summary -------------------------------------------------------------------------

save() {
    printf -v "$1" '%s' "$2"
    printf '%s=%q\n' "$1" "$2" >>"$STATE"
}

if [[ -z "${CHAIN_DEVICE:-}" ]]; then
    save CHAIN_DEVICE "$DEVICE"
elif [[ "$CHAIN_DEVICE" != "$DEVICE" ]]; then
    die "$OUT belongs to the $CHAIN_DEVICE chain, not $DEVICE"
fi

render_summary() {
    {
        printf '# A8 v2 (%s)\n\ncommit %s, cache %s, threads %s\n\n## Notes\n\n' \
            "${DEVICE^^}" "$HEAD_COMMIT" "$CACHE_DIR" "$THREADS"
        cat "$NOTES"
        printf '\n## Steps\n\n| n | name | exit | wall_s | log |\n|---|---|---|---|---|\n'
        sed 's/\t/ | /g; s/^/| /; s/$/ |/' "$STEPS"
    } >"$SUMMARY.tmp"
    mv "$SUMMARY.tmp" "$SUMMARY"
}

note() {
    printf -- '- %s\n' "$*" >>"$NOTES"
    render_summary
}

# note_block TITLE: stdin as its own section of the notes.
note_block() {
    { printf '\n### %s\n\n' "$1"; cat; printf '\n'; } >>"$NOTES"
    render_summary
}

begin() {
    CUR_N="$1"
    CUR_NAME="$2"
    CUR_START=$SECONDS
    CALL_K=0
    LAST_LOG=""
    echo "== step $CUR_N: $CUR_NAME"
    # A killed rerun must not leave an earlier 0 as the step's last row.
    row running
}

row() {
    printf '%s\t%s\t%s\t%s\t%s\n' "$CUR_N" "$CUR_NAME" "$1" "$((SECONDS - CUR_START))" \
        "${LAST_LOG:--}" >>"$STEPS"
    render_summary
}

stop() {
    row 2
    die "step $CUR_N ($CUR_NAME): $*"
}

# fatal TOLERATED CODE: whether CODE stops the step (75, the busy lock, is never tolerated).
fatal() {
    [[ "$2" -eq 75 || ("$2" -ne 0 && "$2" -ne "$1") ]]
}

# run_logged TOLERATED -- CMD...: CMD with its output also in a log of the step; exits with its
# code if fatal.
run_logged() {
    local tolerated="$1"
    shift 2
    CALL_K=$((CALL_K + 1))
    LAST_LOG="$LOGS/$CUR_N-$CUR_NAME-$CALL_K.log"
    printf '$ %s\n' "$*" >"$LAST_LOG"
    set +e
    "$@" 2>&1 | tee -a "$LAST_LOG"
    local code="${PIPESTATUS[0]}"
    set -e
    if fatal "$tolerated" "$code"; then
        row "$code"
        echo "step $CUR_N ($CUR_NAME) failed ($code): $*; log $LAST_LOG" >&2
        exit "$code"
    fi
    if [[ "$code" -ne 0 ]]; then
        note "step $CUR_N ($CUR_NAME): some clips failed (exit $code), see $LAST_LOG"
    fi
}

call() {
    local tolerated="$1"
    shift 2
    run_logged "$tolerated" -- pv "$@"
}

# The run directory the last call printed.
run_of() {
    local run_dir
    run_dir="$(sed -n '/^run: /{s///p;q}' "$LAST_LOG")"
    [[ -n "$run_dir" && -d "$run_dir" ]] || stop "no run directory in $LAST_LOG"
    printf '%s\n' "$run_dir"
}

ids_words() {
    tr -s '[:blank:]' '\n' <"$1" | sed '/^$/d'
}

# first_extracted IDS ALLOWED EXCLUDED: the first shard id in ALLOWED and not in EXCLUDED.
first_extracted() {
    ids_words "$1" | grep -Fx -f "$2" | grep -Fvx -f "$3" | head -n 1 || true
}

# --- startup checks ----------------------------------------------------------------------------

# The PISA selection every step reads must be within what the shards extracted, or step 2 would
# run DETECT..DEPTH on the rest; without PHYSVERIFY_A8V2_PISA_MAX_CLIPS it must be all of it.
check_pisa() {
    local ids="$SHARDS/pisa_real_full.ids"
    [[ -f "$ids" ]] || die "no $ids"
    query clips pisa_real full >"$OUT/pisa_selected.txt" || die "cannot list the PISA selection"
    ids_words "$ids" | sort -u >"$OUT/pisa_extracted.txt"
    local selected extracted missing
    selected="$(grep -c . "$OUT/pisa_selected.txt" || true)"
    extracted="$(grep -c . "$OUT/pisa_extracted.txt" || true)"
    missing="$(grep -Fvxc -f "$OUT/pisa_extracted.txt" "$OUT/pisa_selected.txt" || true)"
    [[ "$selected" -gt 0 ]] || die "the PISA selection is empty"
    [[ "$missing" -eq 0 ]] || die "$missing selected PISA clips were not extracted by the shards"
    if [[ -z "${PHYSVERIFY_A8V2_PISA_MAX_CLIPS:-}" && "$selected" -ne "$extracted" ]]; then
        die "PISA selects $selected clips, the shards extracted $extracted:" \
            "set PHYSVERIFY_A8V2_PISA_MAX_CLIPS"
    fi
    echo "PISA: $selected clips selected, $extracted extracted"
}

preflight_clip() {
    local override="$1" dataset="$2" ids="$3" split="${4:-}"
    if [[ -n "$override" ]]; then
        printf '%s\n' "$override"
        return
    fi
    local args=(clips "$dataset" full)
    [[ -z "$split" ]] || args+=(--split "$split")
    query "${args[@]}" >"$OUT/preflight_allowed.txt" || die "cannot list $dataset clips"
    : >"$OUT/preflight_excluded.txt"
    if [[ "$dataset" == kubric_synthetic ]]; then
        query rerendered "$MANIFEST" --missing-ok >"$OUT/preflight_excluded.txt" \
            || die "cannot read $MANIFEST"
    fi
    [[ -f "$ids" ]] || die "no $ids"
    first_extracted "$ids" "$OUT/preflight_allowed.txt" "$OUT/preflight_excluded.txt"
}

preflight_verify() {
    local dataset="$1" clip="$2"
    [[ -n "$clip" ]] || stop "no extracted $dataset clip to preflight"
    call 1 -- verify --dataset "$dataset" --level perceived --subset full --clips "$clip"
    local run errors
    run="$(run_of)"
    # verify exits 0 on an uncached clip: its rows abstain with the error in checks.jsonl.
    errors="$(query run-errors "$run" 2>&1)" \
        || stop "$dataset clip $clip failed or is not cached for --device $PV_DEVICE:"$'\n'"$errors"
    note "preflight: $dataset $clip cached, run $run"
}

# The cost of steps 2 and 4 from one calib clip whose variants were never computed: none once
# step 2 has begun, and each clip is timed at most once.
estimate() {
    if [[ "$FROM" -gt 2 || -n "${STAGES_BEGUN:-}" ]]; then
        note "estimate: none, step 2 has already begun and its variants are cached"
        return
    fi
    query clips kubric_synthetic full --split calib >"$OUT/kubric_calib.txt" \
        || die "cannot list the kubric_synthetic calib clips"
    query rerendered "$MANIFEST" --missing-ok >"$OUT/estimate_excluded.txt" \
        || die "cannot read $MANIFEST"
    tr -s ' ' '\n' <<<"${TIMED_CLIPS:-}" >>"$OUT/estimate_excluded.txt"
    local clip calib start seconds
    clip="$(first_extracted "$SHARDS/kubric_full.ids" "$OUT/kubric_calib.txt" \
        "$OUT/estimate_excluded.txt")"
    if [[ -z "$clip" ]]; then
        note "estimate: none, every extracted calib clip has been timed before"
        return
    fi
    save TIMED_CLIPS "${TIMED_CLIPS:+$TIMED_CLIPS }$clip"
    calib="$(grep -c . "$OUT/kubric_calib.txt" || true)"
    start=$SECONDS
    call 1 -- diagnose stages "${KS[@]}" --clips "$clip"
    seconds=$((SECONDS - start))
    note "estimate: diagnose stages took ${seconds}s on calib clip $clip, not computed before," \
        "model loading included; steps 2 and 4 each take about $((seconds * calib))s on the" \
        "$calib calib clips (one-clip estimate, high by the load time)"
}

# The kubric clip is a calib clip outside the migration's re-renders (their cache is purged).
preflight_clip_of() {
    case "$1" in
        kubric_synthetic) preflight_clip "${PHYSVERIFY_A8V2_PREFLIGHT_KUBRIC:-}" "$1" \
            "$SHARDS/kubric_full.ids" calib ;;
        physics_iq) preflight_clip "${PHYSVERIFY_A8V2_PREFLIGHT_PHYSICS_IQ:-}" "$1" \
            "$SHARDS/$1_full.ids" ;;
        pisa_real) preflight_clip "${PHYSVERIFY_A8V2_PREFLIGHT_PISA:-}" "$1" \
            "$SHARDS/$1_full.ids" ;;
        *) die "no preflight clip for $1" ;;
    esac
}

# A wrong cache would recompute every clip in extract run; one cached clip per dataset proves it.
# Only the range's datasets: the others may still be under their shards.
preflight() {
    begin pre preflight
    local dataset clip picked=()
    for dataset in $RANGE_DATASETS; do
        clip="$(preflight_clip_of "$dataset")"
        picked+=("$dataset" "$clip")
    done
    local i
    for ((i = 0; i < ${#picked[@]}; i += 2)); do
        preflight_verify "${picked[$i]}" "${picked[$((i + 1))]}"
    done
    estimate
    row 0
}

# --- steps -------------------------------------------------------------------------------------

step_0() {
    if query rerendered "$MANIFEST" >/dev/null 2>&1 && [[ -f "$AUDIT" ]]; then
        note "migration $MIGRATION: already complete ($MANIFEST, $AUDIT), not rerun"
        return
    fi
    # The migration rewrites the shared dataset and purges its cache under the CPU chain.
    [[ "$DEVICE" == cpu ]] \
        || stop "migration $MIGRATION is incomplete: it is shared, run step 0 of the cpu chain"
    run_logged 0 -- env -u PHYSVERIFY_LOCK_HELD \
        PHYSVERIFY_MIGRATE_ROOT="$H200_ROOT" PHYSVERIFY_MIGRATE_VERSION="$VERSION" \
        PHYSVERIFY_MIGRATE_NAME="$MIGRATION" PHYSVERIFY_MIGRATE_SUBSETS=full \
        PHYSVERIFY_MIGRATE_PYTHON="$PY" PHYSVERIFY_MIGRATE_LOCK="$PHYSVERIFY__RUNTIME__HEAVY_LOCK" \
        bash "$REPO_ROOT/scripts/h200/migrate_velocity.sh"
    note "migration $MIGRATION: audit $AUDIT"
}

step_1() {
    query rerendered "$MANIFEST" >"$OUT/rerendered.txt" || stop "no completed $MANIFEST"
    call 1 -- extract run "${KS[@]}"
    local failed
    failed="$(sed -n 's/^FAILED \([^ ]*\) at .*/\1/p' "$LAST_LOG" \
        | grep -Fxc -f "$OUT/rerendered.txt" || true)"
    note "re-rendered clips: $(grep -c . "$OUT/rerendered.txt" || true), failed: ${failed:-0}"
}

# Round-robin shards of the full selection, each a `diagnose stages` without the profile; the heavy
# cache keys do not depend on the error profile and its writes are atomic, so they can overlap.
# Like the profile call after them, they tolerate exit 1 (failed clips) and nothing else.
diag_shards() {
    query clips kubric_synthetic full >"$OUT/diag_clips.txt" \
        || stop "cannot list the kubric_synthetic clips"
    local i id log pids=() logs=() clips=()
    for ((i = 0; i < DIAG_SHARDS; i++)); do
        clips=()
        while IFS= read -r id; do
            clips+=(--clips "$id")
        done < <(sed -n "$((i + 1))~${DIAG_SHARDS}p" "$OUT/diag_clips.txt")
        [[ ${#clips[@]} -gt 0 ]] || continue
        log="$LOGS/$CUR_N-$CUR_NAME-shard-$i.log"
        (
            THREADS="$DIAG_THREADS"
            PV_NICE=(nice -n 5)
            pv --no-lock diagnose stages "${KS[@]}" "${clips[@]}"
        ) >"$log" 2>&1 &
        pids+=("$!")
        logs+=("$log")
    done
    [[ ${#pids[@]} -gt 0 ]] || stop "no kubric_synthetic clips to shard"
    local code failed=0
    for i in "${!pids[@]}"; do
        code=0
        wait "${pids[$i]}" || code=$?
        note "step 2 shard $i ($DIAG_THREADS threads): exit $code, log ${logs[$i]}"
        ! fatal 1 "$code" || failed=$((failed + 1))
    done
    [[ "$failed" -eq 0 ]] || stop "$failed of ${#pids[@]} diagnose shards failed"
}

step_2() {
    save STAGES_BEGUN 1
    [[ -z "$DIAG_SHARDS" ]] || diag_shards
    call 1 -- diagnose stages "${KS[@]}" --write-profile
    local run
    run="$(run_of)"
    save STAGES_RUN "$run"
    note "stage errors: $STAGES_RUN"
}

step_3() {
    call 1 -- extract run "${KS[@]}"
    call 1 -- extract run --dataset physics_iq --subset full
    call 1 -- extract run --dataset pisa_real --subset full
}

step_4() {
    call 0 -- calibrate stage-sigma "${KS[@]}"
}

step_5() {
    call 0 -- calibrate fit "${KS[@]}" --level both
}

step_6() {
    call 0 -- calibrate evaluate "${KS[@]}" --real
    if grep -q '^real dataset .* skipped' "$LAST_LOG"; then
        stop "a real dataset was skipped (see $LAST_LOG)"
    fi
    local evaluation checked
    evaluation="$(sed -n '/^evaluation: /{s///p;q}' "$LAST_LOG")"
    [[ -n "$evaluation" && -f "$evaluation" ]] || stop "no evaluation in $LAST_LOG"
    save EVAL_JSON "$evaluation"
    checked="$(query evaluation "$EVAL_JSON")" || stop "the evaluation failed its checks"
    note_block "Evaluation ($EVAL_JSON)" <<<"$checked"
}

step_7() {
    call 0 -- abstain fit
    call 0 -- abstain curve
}

# Rule 1.2: the same clips verified with the precise profile's detector and with the tiny one.
step_8() {
    local clips_file="$OUT/r2_clips.txt"
    call 0 -- exclusions r2-clips --out "$clips_file"
    local clips=() id base run
    while IFS= read -r id; do
        [[ -z "$id" ]] || clips+=(--clips "$id")
    done <"$clips_file"
    [[ ${#clips[@]} -gt 0 ]] || stop "no R2 clips in $clips_file"
    call 1 -- verify "${KS[@]}" --level perceived --policy raw "${clips[@]}"
    base="$(run_of)"
    save R2_BASE_RUN "$(basename "$base")"
    note "R2 base arm: $base, checks_hash $(cat "$base/checks_hash.txt" 2>/dev/null || echo '?')"
    local tiny=(--set extraction.plan.detect=grounding_dino_tiny)
    call 1 -- "${tiny[@]}" extract run "${KS[@]}" "${clips[@]}"
    call 1 -- "${tiny[@]}" verify "${KS[@]}" --level perceived --policy raw "${clips[@]}"
    run="$(run_of)"
    save R2_TINY_RUN "$(basename "$run")"
    note "R2 tiny arm: $run, checks_hash $(cat "$run/checks_hash.txt" 2>/dev/null || echo '?')"
    cmp -s "$base/checks_hash.txt" "$run/checks_hash.txt" \
        || stop "the R2 arms ran with different check parameters (checks_hash.txt)"
}

step_9() {
    [[ -n "${EVAL_JSON:-}" ]] || stop "no evaluation recorded: rerun from 6"
    [[ -n "${R2_BASE_RUN:-}" && -n "${R2_TINY_RUN:-}" ]] || stop "no R2 arms recorded: rerun from 8"
    local args=(exclusions decide --evaluation "$EVAL_JSON" --r2-base-run "$R2_BASE_RUN"
        --r2-tiny-run "$R2_TINY_RUN")
    [[ "${PHYSVERIFY_A8V2_DECIDE_REPLACE:-0}" != 1 ]] || args+=(--replace)
    call 0 -- "${args[@]}"
    call 0 -- exclusions show
    local sha
    sha="$(sed -n 's/^record: .* sha256 \([0-9a-f]*\)$/\1/p' "$LAST_LOG")"
    [[ -n "$sha" ]] || stop "no record sha256 in $LAST_LOG"
    save RECORD_SHA "$sha"
    note "exclusion record sha256 $RECORD_SHA"
}

step_10() {
    call 0 -- size eval
}

step_11() {
    [[ -n "${STAGES_RUN:-}" ]] || stop "no stage errors recorded: rerun from 2"
    local rows
    rows="$(query gravity "$STAGES_RUN/diagnosis/stages.json")" || stop "no gravity rows"
    note_block "LIFT gravity angle" <<<"$rows"
}

# A tolerated exit 1 is recorded as 0 (run_logged notes it), so any other code is a stop.
require_earlier_steps() {
    local k code key
    for ((k = 0; k < FROM; k++)); do
        code="$(awk -F '\t' -v k="$k" '$1 == k { c = $3 } END { print c }' "$STEPS")"
        [[ "$code" == 0 ]] \
            || die "step $k has not completed (last exit '${code:-none}'): rerun from $k"
        for key in ${STEP_SAVES[$k]:-}; do
            [[ -n "${!key:-}" ]] || die "step $k recorded no $key in $STATE: rerun from $k"
        done
    done
}

# --- main --------------------------------------------------------------------------------------

require_earlier_steps
# Steps 3, 6 and 7 read the PISA selection.
! in_range pisa_real || [[ "$FROM" -gt 7 ]] || check_pisa
[[ "$FROM" -gt 8 ]] || preflight
for ((n = FROM; n <= TO; n++)); do
    busy_guard
    case "$n" in
        0) begin 0 migrate ;;
        1) begin 1 extract-synthetic ;;
        2) begin 2 stages ;;
        3) begin 3 relift ;;
        4) begin 4 stage-sigma ;;
        5) begin 5 fit ;;
        6) begin 6 evaluate ;;
        7) begin 7 abstain ;;
        8) begin 8 r2-arms ;;
        9) begin 9 decide ;;
        10) begin 10 size-eval ;;
        11) begin 11 gravity ;;
    esac
    "step_$n"
    row 0
done
if [[ "$TO" -lt "$LAST_STEP" ]]; then
    note "stopped after step $TO as requested; resume with run_a8v2.sh $((TO + 1))"
    echo "A8 v2: stopped after step $TO as requested (resume with run_a8v2.sh $((TO + 1)))"
fi
echo "A8 v2: $SUMMARY"
