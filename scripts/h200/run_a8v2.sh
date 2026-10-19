#!/usr/bin/env bash
# A8 v2 on the H200 CPU (ADR 0015, "Corrida CPU"): the pair-collision migration, the stage
# errors and sigma, the threshold fit and evaluation (with the H1 primary scene view, 1.4), the
# abstention, both R2 detector arms, the exclusion record of 1.1, 1.2 and 1.5, then the Objectron
# size evaluation and the LIFT gravity rows (neither feeds the record, so they come after it).
# It consumes the perception cache the CPU shards left (run_real_cpu_shards.sh,
# run_kubric_cpu_shards.sh) and never uses a GPU: every call is --profile precise --device cpu,
# the device the cache keys were computed with.
#
#   bash scripts/h200/run_a8v2.sh [FROM]     FROM: first step to run (0-11, default 0)
#
#   startup  no running extract/dataset job; HEAD is PHYSVERIFY_A8V2_EXPECT_COMMIT with no
#            uncommitted change under src, configs or scripts; paths.cache_dir is
#            PHYSVERIFY_A8V2_CACHE_DIR; the PISA selection is within what the shards extracted
#   pre      (FROM <= 8) one cached clip per dataset verified (any error in its checks.jsonl
#            stops), so a wrong cache stops here and not after days of recomputation; before step 2
#            has begun, one timed `diagnose stages` calib clip not timed before (cost of steps 2, 4)
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
# stage the shards never ran) on the CPU under the lazy heavy lock; the preflight does not cover
# them, it only times one clip of them and writes the estimate to the summary.
# Everything lands in $PHYSVERIFY_A8V2_DIR (default $H200_ROOT/a8v2): logs/, steps.tsv,
# summary.md and state.env (run ids later steps need, so FROM resumes). Exit codes: 0 ok, 2 a
# check failed, otherwise the failing command's own (1 failed clips where not tolerated, 75 lock).
#
# Required:
#   PHYSVERIFY_A8V2_EXPECT_COMMIT   prefix (>= 7 chars) of the HEAD this checkout must be at
#   PHYSVERIFY_A8V2_CACHE_DIR       the cache root the shards filled; doctor must report it
# Optional:
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
FROM="${1:-0}"
[[ "$FROM" =~ ^(0|[1-9][0-9]*)$ && "$FROM" -le "$LAST_STEP" ]] \
    || die "FROM must be a step number 0-$LAST_STEP, got '$FROM'"
BUSY_JOBS='physverify(\.cli\.main)?.*(extract run|dataset (build|migrate))'

# The shards ran with --no-lock, so the heavy lock cannot tell whether they are done.
procs="$(ps -eo pid=,args=)"
busy="$(grep -E "$BUSY_JOBS" <<<"$procs" | grep -vE "^ *$$ " || true)"
[[ -z "$busy" ]] || die "extraction or dataset jobs are still running:"$'\n'"$busy"

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
export CUDA_VISIBLE_DEVICES=""
unset PHYSVERIFY_LOCK_HELD
PV_DEVICE=cpu
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

DOCTOR="$(pv doctor)" || die "physverify doctor failed"
CACHE_DIR="$(doctor_value paths.cache_dir)"
[[ -n "$CACHE_DIR" && "$(realpath -m "$CACHE_DIR")" == "$(realpath -m "$EXPECT_CACHE")" ]] \
    || die "doctor reports paths.cache_dir '$CACHE_DIR', not the shards' $EXPECT_CACHE"
# The profile of 1.19: the precise one, never a recalibrated copy.
ERROR_PROFILE="$(doctor_value paths.error_profile)"
[[ "$(basename "$ERROR_PROFILE")" == stage_errors.precise.json ]] \
    || die "doctor reports paths.error_profile '$ERROR_PROFILE', not stage_errors.precise.json"
SYNTH_DIR="$(doctor_value paths.synthetic_dir)"
VERSION="${PHYSVERIFY_A8V2_VERSION:-v1-precise}"
MIGRATION=pair-collision
MANIFEST="$SYNTH_DIR/$VERSION/migrations/$MIGRATION.json"
AUDIT="$(doctor_value paths.data_dir)/runs/migrate-$MIGRATION-$VERSION/audit.json"
SHARDS="${PHYSVERIFY_A8V2_SHARDS:-$H200_ROOT/logs/shards}"

OUT="${PHYSVERIFY_A8V2_DIR:-$H200_ROOT/a8v2}"
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

render_summary() {
    {
        printf '# A8 v2 (CPU)\n\ncommit %s, cache %s, threads %s\n\n## Notes\n\n' \
            "$HEAD_COMMIT" "$CACHE_DIR" "$THREADS"
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

# run_logged TOLERATED -- CMD...: CMD with its output also in a log of the step; exits with its
# code unless that is 0 or TOLERATED (75, the busy lock, is never tolerated).
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
    if [[ "$code" -eq 75 || ("$code" -ne 0 && "$code" -ne "$tolerated") ]]; then
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
        || stop "$dataset clip $clip failed or is not cached for --device cpu:"$'\n'"$errors"
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

# A wrong cache would recompute every clip in extract run; one cached clip per dataset proves it.
# The kubric clip is a calib clip outside the migration's re-renders (their cache is purged).
preflight() {
    begin pre preflight
    local kubric physics pisa
    kubric="$(preflight_clip "${PHYSVERIFY_A8V2_PREFLIGHT_KUBRIC:-}" kubric_synthetic \
        "$SHARDS/kubric_full.ids" calib)"
    physics="$(preflight_clip "${PHYSVERIFY_A8V2_PREFLIGHT_PHYSICS_IQ:-}" physics_iq \
        "$SHARDS/physics_iq_full.ids")"
    pisa="$(preflight_clip "${PHYSVERIFY_A8V2_PREFLIGHT_PISA:-}" pisa_real \
        "$SHARDS/pisa_real_full.ids")"
    preflight_verify kubric_synthetic "$kubric"
    preflight_verify physics_iq "$physics"
    preflight_verify pisa_real "$pisa"
    estimate
    row 0
}

# --- steps -------------------------------------------------------------------------------------

step_0() {
    if query rerendered "$MANIFEST" >/dev/null 2>&1 && [[ -f "$AUDIT" ]]; then
        note "migration $MIGRATION: already complete ($MANIFEST, $AUDIT), not rerun"
        return
    fi
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
        [[ "$code" -eq 0 ]] || failed=$((failed + 1))
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

# --- main --------------------------------------------------------------------------------------

[[ "$FROM" -gt 6 ]] || check_pisa
[[ "$FROM" -gt 8 ]] || preflight
for ((n = FROM; n <= LAST_STEP; n++)); do
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
echo "A8 v2: $SUMMARY"
