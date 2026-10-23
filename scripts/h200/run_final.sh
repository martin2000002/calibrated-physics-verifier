#!/usr/bin/env bash
# The sealed final evaluation on the H200, in a clean checkout at the frozen commit of
# $F/freeze.json (run_freeze.sh). Every call is --profile precise --device cpu with the frozen
# artifacts of $F/calibration.
#
#   bash scripts/h200/run_final.sh seed            the preregistered seed is the freeze's
#   bash scripts/h200/run_final.sh render SET      kubric_fresh | kubric_fresh_ood (run_kubric.sh)
#   bash scripts/h200/run_final.sh fetch SET       physics_iq_views | objectron_static
#   bash scripts/h200/run_final.sh extract SET     any set but real_new (exempt: prints no verdict)
#   bash scripts/h200/run_final.sh look SET [--retry]   the one look at a sealed set
#   bash scripts/h200/run_final.sh v1arm SET       kubric_fresh | kubric_synthetic_eval | real_new
#
# Sets, the specs a look unseals and what it runs (calibrate evaluate --artifact $F/calibration/
# calibration.json ...):
#   kubric_synthetic_eval  kubric_synthetic:eval      --dataset kubric_synthetic --subset eval --real
#   kubric_fresh           kubric_fresh               --dataset kubric_fresh --subset eval --no-real
#   kubric_fresh_ood       kubric_fresh_ood           as kubric_fresh (its own OOD render)
#   kubric_fresh_blind     kubric_fresh_blind         as kubric_fresh (kubric_fresh's render)
#   real_new               physics_iq_views,objectron_static
#                          --dataset kubric_synthetic --subset calib --real-dataset physics_iq_views
#                          --real-dataset objectron_static (each extracted as its own set first)
#
# Every command, before any physverify call (exit 2): HEAD is PHYSVERIFY_FINAL_EXPECT_COMMIT and the
# freeze's commit, with nothing uncommitted or untracked under src, configs, scripts, annotations,
# pyproject.toml or uv.lock; no configs/local.toml, no PHYSVERIFY_CONFIG_DIR, no PHYSVERIFY__* variable but the paths exported
# here; no PHYSVERIFY_UNSEAL or PHYSVERIFY_UNSEAL_REPEAT from the caller (this script sets them for
# a look and a v1 arm only); freeze.json of device cpu and profile precise with its calibration
# directory $F/calibration and its frozen files unchanged (freeze_query artifacts);
# PHYSVERIFY_FINAL_SEED is the freeze's preregistered seed; $F/run-final/state.env of the same
# commit and dry-run mode; for any command but seed, the seed command has run. Then doctor must
# report the committed configs, cpu, precise, the cache and the calibration in $F.
#
# Fresh data: extract refuses a set whose cache directory ($CACHE/<dataset>) exists before its first
# extraction (the marker $F/run-final/extract-SET.started tells a resume), and every Kubric command
# needs the render's plan.json of the seed (freeze_query plan). Render resumes the same version
# and seed only, never after a look; the exported share of the plan is noted against 90 %. One
# look: look refuses a set any ledger line covers; --retry passes once, when that line's run wrote
# no evaluation, with PHYSVERIFY_UNSEAL_REPEAT=1; a missing ledger stops render, look and v1arm (it
# is created once by hand, never by a look). Fixed downloads: fetch writes the sha256 manifest of
# the downloaded files; extract and look check it, and fetch is refused once extraction has started.
# A set's commands hold $F/run-final/<command>-<set>.lock; extract also holds its render's and its
# fetch's lock, and look its extractions' locks.
#
# Required:
#   PHYSVERIFY_FINAL_EXPECT_COMMIT  prefix (>= 7 chars) of the freeze commit, the HEAD this
#                                   checkout is at
#   PHYSVERIFY_FINAL_SEED           the preregistered seed
# Optional (defaults in brackets):
#   PHYSVERIFY_FINAL_DIR            $F [$H200_ROOT/final]
#   PHYSVERIFY_FINAL_FREEZE         the freeze manifest [$F/freeze.json]
#   PHYSVERIFY_FINAL_DATA           paths.data_dir [$H200_ROOT/kubric-data]
#   PHYSVERIFY_FINAL_CACHE_DIR      paths.cache_dir [$DATA/cache]
#   PHYSVERIFY_FINAL_PYTHON         interpreter [/home/mmonterot/dev/venv/bin/python]
#   PHYSVERIFY_FINAL_VENV           UV_PROJECT_ENVIRONMENT of run_kubric.sh [the interpreter's venv]
#   PHYSVERIFY_FINAL_THREADS        runtime.threads [32]
#   PHYSVERIFY_FINAL_EXTRACT_LIMIT  extract only the first N clips (to measure the pace) [all]
#   PHYSVERIFY_FINAL_EXTRACT_SHARDS --no-lock extract shards under nice before the full pass [none]
#   PHYSVERIFY_FINAL_SHARD_THREADS  runtime.threads of each shard [THREADS / SHARDS]
#   PHYSVERIFY_FINAL_DRYRUN         1 for a dry-run rehearsal: $F, the data, the cache and the ledger
#                                   (PHYSVERIFY__SEALED__LEDGER, then required) under
#                                   $H200_ROOT/scratch; PHYSVERIFY__KUBRIC_FRESH__*,
#                                   PHYSVERIFY__PHYSICS_IQ_VIEWS__* and PHYSVERIFY__OBJECTRON_STATIC__*
#                                   allowed; the seed must differ from the freeze's (and from that
#                                   of $H200_ROOT/final/freeze.json, when it exists); never
#                                   kubric_synthetic_eval; physics_iq_views and real_new need
#                                   PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS='["center"]', objectron_static
#                                   and real_new PHYSVERIFY__OBJECTRON_STATIC__SPLIT=train, so that a
#                                   rehearsal never reads a sealed population [0]
# v1arm (the v1 commit's checks with the base iteration's calibration, after the look):
#   PHYSVERIFY_FINAL_V1_CHECKOUT    clean checkout at the freeze's v1.commit (the v1 commit)
#   PHYSVERIFY_FINAL_V1_BASE        the base iteration directory (its calibration.json is the
#                                   freeze's v1.calibration_sha256)
#   PHYSVERIFY_FINAL_OVERLAY_CLI    the overlay CLI script the v1 arm runs (kept outside the repo)
#   PHYSVERIFY_FINAL_V1_ARM         the arm's directory [$F/v1arm]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steps.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_guards.sh"
cd "$REPO_ROOT"

USAGE="usage: run_final.sh seed | render SET | fetch SET | extract SET | look SET [--retry] | v1arm SET"
CMD="${1:-}"
SET="${2:-}"
RETRY=""
case "$CMD" in
    seed) [[ $# -eq 1 ]] || die "$USAGE" ;;
    render | fetch | extract | v1arm) [[ $# -eq 2 ]] || die "$USAGE" ;;
    look)
        [[ $# -eq 2 || ($# -eq 3 && "$3" == --retry) ]] || die "$USAGE"
        RETRY="${3:-}"
        ;;
    *) die "$USAGE" ;;
esac

# describe SET: DS and SUBSET it is extracted as, the SPECS a look unseals (comma separated), the
# sets whose extraction the look NEEDS, the RENDER set of a Kubric set and the LOOK arguments.
describe() {
    DS="$1"
    SUBSET=eval
    SPECS="$1"
    NEEDS="$1"
    RENDER=""
    LOOK=(--dataset "$1" --subset eval --no-real)
    case "$1" in
        kubric_synthetic_eval)
            DS=kubric_synthetic
            SPECS=kubric_synthetic:eval
            LOOK=(--dataset kubric_synthetic --subset eval --real)
            ;;
        kubric_fresh | kubric_fresh_ood) RENDER="$1" ;;
        kubric_fresh_blind) RENDER=kubric_fresh ;;
        physics_iq_views | objectron_static)
            SUBSET=full
            LOOK=()
            ;;
        real_new)
            DS=""
            SPECS=physics_iq_views,objectron_static
            NEEDS="physics_iq_views objectron_static"
            LOOK=(--dataset kubric_synthetic --subset calib --real-dataset physics_iq_views
                --real-dataset objectron_static)
            ;;
        *) die "unknown set '$1': kubric_synthetic_eval, kubric_fresh, kubric_fresh_ood," \
            "kubric_fresh_blind, physics_iq_views, objectron_static or real_new" ;;
    esac
}

[[ "$CMD" == seed ]] || describe "$SET"
case "$CMD" in
    render) [[ "$SET" == kubric_fresh || "$SET" == kubric_fresh_ood ]] \
        || die "only kubric_fresh and kubric_fresh_ood are rendered (kubric_fresh_blind reuses kubric_fresh)" ;;
    fetch) [[ "$SET" == physics_iq_views || "$SET" == objectron_static ]] \
        || die "only physics_iq_views and objectron_static are fetched" ;;
    extract) [[ -n "$DS" ]] || die "real_new is extracted as physics_iq_views and objectron_static" ;;
    look) [[ ${#LOOK[@]} -gt 0 ]] || die "$SET is looked at with the other new real set: look real_new" ;;
    v1arm) [[ "$SET" =~ ^(kubric_fresh|kubric_synthetic_eval|real_new)$ ]] \
        || die "the v1 arm runs on kubric_fresh, kubric_synthetic_eval and real_new only" ;;
esac

source_h200_env
F="${PHYSVERIFY_FINAL_DIR:-$H200_ROOT/final}"
FREEZE="${PHYSVERIFY_FINAL_FREEZE:-$F/freeze.json}"
DATA="${PHYSVERIFY_FINAL_DATA:-$H200_ROOT/kubric-data}"
CACHE="${PHYSVERIFY_FINAL_CACHE_DIR:-$DATA/cache}"
CAL="$F/calibration"
SCRATCH="$H200_ROOT/scratch"
DRYRUN="${PHYSVERIFY_FINAL_DRYRUN:-0}"
[[ "$DRYRUN" == 0 || "$DRYRUN" == 1 ]] || die "PHYSVERIFY_FINAL_DRYRUN must be 0 or 1"
OWN=" PATHS__DATA_DIR PATHS__CACHE_DIR PATHS__WEIGHTS_DIR PATHS__VENDOR_DIR PATHS__HF_HOME
    PATHS__CALIBRATION_DIR PATHS__RUNS_DIR RUNTIME__HEAVY_LOCK "
if [[ "$DRYRUN" == 1 ]]; then
    [[ -n "${PHYSVERIFY__SEALED__LEDGER:-}" ]] \
        || die "a dry run needs its own ledger: set PHYSVERIFY__SEALED__LEDGER under $SCRATCH"
    for dir in "$F" "$DATA" "$CACHE" "$PHYSVERIFY__SEALED__LEDGER"; do
        under "$dir" "$SCRATCH" || die "a dry run writes only under $SCRATCH, not $dir"
    done
    refuse_foreign_config run_final.sh "$OWN" \
        '^PHYSVERIFY__(SEALED__LEDGER|KUBRIC_FRESH__[A-Z_]+|PHYSICS_IQ_VIEWS__[A-Z_]+|OBJECTRON_STATIC__[A-Z_]+)$'
else
    for dir in "$F" "$DATA" "$CACHE"; do
        ! near "$dir" "$SCRATCH" || die "$dir is under $SCRATCH: only a dry run writes there"
    done
    refuse_foreign_config run_final.sh "$OWN"
fi
for name in PHYSVERIFY_UNSEAL PHYSVERIFY_UNSEAL_REPEAT; do
    [[ -z "${!name+set}" ]] || die "$name is set: run_final.sh sets it for a look and a v1 arm" \
        "only; unset it"
done
refuse_local_config run_final.sh
require_checkout PHYSVERIFY_FINAL_EXPECT_COMMIT
SEED="${PHYSVERIFY_FINAL_SEED:-}"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "set PHYSVERIFY_FINAL_SEED to the preregistered seed"
PY="${PHYSVERIFY_FINAL_PYTHON:-/home/mmonterot/dev/venv/bin/python}"
[[ -x "$PY" ]] || die "no interpreter at $PY (PHYSVERIFY_FINAL_PYTHON)"
VENV="${PHYSVERIFY_FINAL_VENV:-$(dirname "$(dirname "$PY")")}"
THREADS="${PHYSVERIFY_FINAL_THREADS:-32}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || die "PHYSVERIFY_FINAL_THREADS must be a positive integer"
[[ -f "$FREEZE" ]] || die "no freeze manifest at $FREEZE (run_freeze.sh writes it)"
export CUDA_VISIBLE_DEVICES=""
PV_DEVICE=cpu

export PHYSVERIFY__PATHS__DATA_DIR="$DATA"
export PHYSVERIFY__PATHS__CACHE_DIR="$CACHE"
export PHYSVERIFY__PATHS__WEIGHTS_DIR="$H200_ROOT/Repo/data/weights"
export PHYSVERIFY__PATHS__VENDOR_DIR="$H200_ROOT/Repo/data/vendor"
export PHYSVERIFY__PATHS__HF_HOME="$H200_ROOT/.cache/huggingface"
export PHYSVERIFY__PATHS__CALIBRATION_DIR="$CAL"
export PHYSVERIFY__PATHS__RUNS_DIR="$F/runs"
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$F/heavy.lock"
unset PHYSVERIFY_LOCK_HELD

# frozen KEY: one value of freeze.json (exit 2 when it has none).
frozen() {
    fq field "$FREEZE" "$1" || die "freeze.json gives no $1"
}

[[ "$(frozen commit)" == "$HEAD_COMMIT" ]] \
    || die "freeze.json is of commit $(frozen commit), HEAD is $HEAD_COMMIT"
[[ "$(frozen device)" == cpu && "$(frozen profile)" == precise ]] \
    || die "freeze.json is of device $(frozen device), profile $(frozen profile): not cpu, precise"
same_path "$(frozen calibration_dir)" "$CAL" \
    || die "freeze.json froze $(frozen calibration_dir), not $CAL"
fq artifacts "$FREEZE" >/dev/null || die "the frozen files changed (freeze_query artifacts)"
# dry_run_guards: a rehearsal never draws a sealed population (the sealed Kubric scenes of the
# frozen seed, kubric_synthetic eval, the left/right Physics-IQ views, the Objectron test videos).
dry_run_guards() {
    local real="$H200_ROOT/final/freeze.json" pair key name want
    [[ "$SEED" != "$(frozen seed)" ]] \
        || die "a dry run with the freeze's seed $SEED would render its sealed scenes; use another"
    if [[ -f "$real" ]] && ! same_path "$real" "$FREEZE"; then
        [[ "$SEED" != "$(fq field "$real" seed)" ]] \
            || die "a dry run with the seed of $real would render its sealed scenes; use another"
    fi
    [[ "$SET" != kubric_synthetic_eval ]] || die "a dry run never touches kubric_synthetic_eval"
    for pair in 'physics_iq_views PHYSICS_IQ_VIEWS__VIEWS ["center"]' \
        'objectron_static OBJECTRON_STATIC__SPLIT train'; do
        read -r key name want <<<"$pair"
        [[ " ${NEEDS:-} " == *" $key "* ]] || continue
        name="PHYSVERIFY__$name"
        [[ "${!name:-}" == "$want" ]] \
            || die "a dry run of $SET needs $name='$want' (the sealed $key stays sealed)"
    done
}

if [[ "$DRYRUN" == 0 ]]; then
    [[ "$SEED" == "$(frozen seed)" ]] \
        || die "PHYSVERIFY_FINAL_SEED=$SEED is not the freeze's seed $(frozen seed)"
else
    dry_run_guards
fi

# A limited extraction (the pace measurement) is its own step: a look needs a full extract.
STEP="$CMD"
[[ "$CMD" != extract || -z "${PHYSVERIFY_FINAL_EXTRACT_LIMIT:-}" ]] || STEP=extract-limited
RUN="$F/run-final"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOGS="$RUN/logs/$STEP${SET:+-$SET}-$STAMP-$$"
STATE="$RUN/state.env"
STEPS="$RUN/steps.tsv"
NOTES="$RUN/notes-$STEP${SET:+-$SET}.md"
SUMMARY="$RUN/summary-$STEP${SET:+-$SET}.md"
SUMMARY_TITLE="Final: $STEP ${SET:-} ${RETRY:-}"
SUMMARY_INFO="commit $HEAD_COMMIT, seed $SEED, dry run $DRYRUN, profile precise, device cpu, cache $CACHE, data $DATA"
mkdir -p "$LOGS"
touch "$STEPS" "$NOTES"

# hold NAME: $RUN/NAME.lock for the rest of the command (exit 2 while another command holds it).
hold() {
    local fd
    exec {fd}>"$RUN/$1.lock"
    flock -n "$fd" || die "$1 is running (another run_final.sh holds $RUN/$1.lock)"
}

# The state is read and first written under its own lock, released before the command runs.
exec {state_fd}>"$RUN/state.lock"
flock "$state_fd"
if [[ -f "$STATE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE"
fi
for pair in "FINAL_COMMIT $HEAD_COMMIT" "FINAL_DRYRUN $DRYRUN"; do
    read -r key value <<<"$pair"
    if [[ -z "${!key:-}" ]]; then
        save "$key" "$value"
    elif [[ "${!key}" != "$value" ]]; then
        die "$F belongs to $key=${!key}, not $value"
    fi
done
if [[ "$CMD" != seed && "${FINAL_SEED:-}" != "$SEED" ]]; then
    die "the seed command has not run with seed $SEED (state.env has '${FINAL_SEED:-}')"
fi
exec {state_fd}>&-

# completed CMD SET: CMD on SET last recorded exit 0 in steps.tsv.
completed() {
    [[ "$(awk -F '\t' -v c="$1" -v s="$2" '$1 == c && $2 == s { x = $3 } END { print x }' \
        "$STEPS")" == 0 ]]
}

# The plan.json of the render a Kubric set reads must be that of the seed.
check_plan() {
    [[ -z "$RENDER" ]] && return 0
    local plan
    plan="$(fq plan "$RENDER" "$SEED")" || stop "the $RENDER render is not the one of seed $SEED"
    note "$plan"
}

raw_dir_of() {
    printf '%s/%s\n' "$(doctor_value paths.raw_dir)" "$1"
}

# check_fetched SET: the downloaded files are those of fetch's manifest.
check_fetched() {
    local manifest="$RUN/fetch-$1.sha256"
    [[ -f "$manifest" ]] || stop "no $manifest: run fetch $1 first"
    (cd "$(raw_dir_of "$1")" && sha256sum --quiet -c "$manifest") \
        || stop "the files of $1 are not those of $manifest"
}

cmd_seed() {
    hold seed
    if [[ "$DRYRUN" == 1 ]]; then
        note "dry run: seed $SEED, not checked against the formula"
    else
        local calibration exclusions computed
        calibration="$(sha_of "$(frozen artifacts.calibration.path)")"
        exclusions="$(sha_of "$(frozen artifacts.exclusions.path)")"
        computed="$(fq seed "$HEAD_COMMIT" "$calibration" "$exclusions")" \
            || stop "cannot compute the seed"
        [[ "$computed" == "$SEED" ]] || stop "the preregistered formula gives $computed, not $SEED"
        note "seed $SEED = int(sha256(commit + calibration sha + exclusions sha)[:8], 16)"
    fi
    save FINAL_SEED "$SEED"
}

cmd_render() {
    hold "render-$SET"
    local covered=("$SET") state planned clips exported share line renv=()
    [[ "$SET" != kubric_fresh ]] || covered+=(kubric_fresh_blind)
    state="$(fq look-state "${covered[@]}")" || stop "a covering ledger line exists: no render after a look"
    note "ledger: $state look at ${covered[*]}"
    if line="$(fq plan "$SET" "$SEED" 2>&1)"; then
        note "resuming the render of seed $SEED: $line"
    elif [[ "$line" != *"render first"* ]]; then
        stop "$line"
    fi
    while IFS= read -r line; do
        renv+=("$line")
    done < <(fq render-env "$SET" "$SEED")
    [[ ${#renv[@]} -gt 0 ]] || stop "no render settings for $SET"
    run_logged 0 -- env "${renv[@]}" UV_PROJECT_ENVIRONMENT="$VENV" PYTHONPATH="$REPO_ROOT/src" \
        PHYSVERIFY_KUBRIC_ENGINE="${PHYSVERIFY_KUBRIC_ENGINE:-udocker}" \
        bash "$REPO_ROOT/scripts/h200/run_kubric.sh" full
    line="$(fq plan "$SET" "$SEED")" || stop "the render left no plan of seed $SEED"
    note "$line"
    planned="$(sed -n 's/.* scenes, \([0-9]*\) clips, .*/\1/p' <<<"$line")"
    [[ "$planned" =~ ^[1-9][0-9]*$ ]] || stop "no planned clip count in: $line"
    clips="$(query clips "$SET" eval)" || stop "cannot list the $SET clips"
    exported="$(grep -c . <<<"$clips" || true)"
    share="$(awk -v e="$exported" -v p="$planned" 'BEGIN { printf "%.1f", 100 * e / p }')"
    note "exported share: $exported of $planned planned clips exported ($share %, threshold 90 %)"
}

cmd_fetch() {
    hold "fetch-$SET"
    hold "extract-$SET"
    [[ ! -e "$RUN/extract-$SET.started" ]] \
        || stop "the extraction of $SET started: its downloaded files are fixed"
    export PHYSVERIFY__OBJECTRON_STATIC__SEED="$SEED"
    call 0 -- download fetch "$SET" --subset full
    unset PHYSVERIFY__OBJECTRON_STATIC__SEED
    local raw manifest="$RUN/fetch-$SET.sha256"
    raw="$(raw_dir_of "$SET")"
    [[ -d "$raw" ]] || stop "fetch left no $raw"
    (cd "$raw" && find . -type f ! -name '*.tmp' -print0 | sort -z | xargs -0 -r sha256sum) \
        >"$manifest.tmp"
    mv "$manifest.tmp" "$manifest"
    note "fetched $SET: $(grep -c . "$manifest") files under $raw, manifest $manifest sha256" \
        "$(sha_of "$manifest")"
}

cmd_extract() {
    hold "extract-$SET"
    [[ -z "$RENDER" ]] || hold "render-$RENDER"
    [[ "$SUBSET" != full ]] || hold "fetch-$SET"
    check_plan
    [[ "$SUBSET" != full ]] || check_fetched "$SET"
    local marker="$RUN/extract-$SET.started"
    if [[ "$SET" != kubric_synthetic_eval && -e "$CACHE/$DS" && ! -e "$marker" ]]; then
        stop "$CACHE/$DS exists before the first extraction of $SET: another render or a dry run" \
            "wrote it"
    fi
    touch "$marker"
    local clips_file="$LOGS/clips.txt" clips=() id
    query clips "$DS" "$SUBSET" >"$clips_file" || stop "cannot list the $DS $SUBSET clips"
    if [[ -n "${PHYSVERIFY_FINAL_EXTRACT_LIMIT:-}" ]]; then
        [[ "$PHYSVERIFY_FINAL_EXTRACT_LIMIT" =~ ^[1-9][0-9]*$ ]] \
            || stop "PHYSVERIFY_FINAL_EXTRACT_LIMIT must be a positive integer"
        sed -i "$((PHYSVERIFY_FINAL_EXTRACT_LIMIT + 1)),\$d" "$clips_file"
        while IFS= read -r id; do
            clips+=(--clips "$id")
        done <"$clips_file"
    fi
    note "extract $DS $SUBSET: $(grep -c . "$clips_file") clips, list sha256 $(sha_of "$clips_file")"
    [[ -z "${PHYSVERIFY_FINAL_EXTRACT_SHARDS:-}" ]] || extract_shards "$clips_file"
    call 1 -- extract run --dataset "$DS" --subset "$SUBSET" "${clips[@]}"
}

# extract_shards CLIPS_FILE: round-robin --no-lock shards under nice that warm the cache before
# the full pass (the cache writes are atomic); like it, they tolerate exit 1 only.
extract_shards() {
    local shards="$PHYSVERIFY_FINAL_EXTRACT_SHARDS" i id log pids=() logs=() clips=()
    [[ "$shards" =~ ^[1-9][0-9]*$ ]] || stop "PHYSVERIFY_FINAL_EXTRACT_SHARDS must be a positive integer"
    local per=$((THREADS / shards))
    local threads="${PHYSVERIFY_FINAL_SHARD_THREADS:-$((per > 0 ? per : 1))}"
    [[ "$threads" =~ ^[1-9][0-9]*$ ]] || stop "PHYSVERIFY_FINAL_SHARD_THREADS must be a positive integer"
    for ((i = 0; i < shards; i++)); do
        clips=()
        while IFS= read -r id; do
            clips+=(--clips "$id")
        done < <(sed -n "$((i + 1))~${shards}p" "$1")
        [[ ${#clips[@]} -gt 0 ]] || continue
        log="$LOGS/shard-$i.log"
        (
            THREADS="$threads"
            PV_NICE=(nice -n 5)
            pv --no-lock extract run --dataset "$DS" --subset "$SUBSET" "${clips[@]}"
        ) >"$log" 2>&1 &
        pids+=("$!")
        logs+=("$log")
    done
    local code failed=0
    for i in "${!pids[@]}"; do
        code=0
        wait "${pids[$i]}" || code=$?
        note "extract shard $i ($threads threads): exit $code, log ${logs[$i]}"
        ! fatal 1 "$code" || failed=$((failed + 1))
    done
    [[ "$failed" -eq 0 ]] || stop "$failed of ${#pids[@]} extract shards failed"
}

cmd_look() {
    hold "look-$SET"
    local need state evaluation ledger
    for need in $NEEDS; do
        hold "extract-$need"
        completed extract "$need" || stop "extract $need has not completed"
        describe "$need"
        check_plan
        [[ "$SUBSET" != full ]] || check_fetched "$need"
    done
    describe "$SET"
    ledger="$(fq ledger)" || stop "the sealed ledger is unreadable or missing (fail closed)"
    note "before: $ledger"
    state="$(fq look-state ${RETRY:+--retry} ${SPECS//,/ })" || stop "one look per set: no look at $SPECS"
    export PHYSVERIFY_UNSEAL="$SPECS"
    [[ "$state" != retry ]] || export PHYSVERIFY_UNSEAL_REPEAT=1
    note "$state look at $SPECS"
    call 0 -- calibrate evaluate --artifact "$CAL/calibration.json" "${LOOK[@]}"
    unset PHYSVERIFY_UNSEAL PHYSVERIFY_UNSEAL_REPEAT
    note "after: $(fq ledger)"
    evaluation="$(evaluation_of "$LAST_LOG")"
    note "evaluation $evaluation sha256 $(sha_of "$evaluation") (report $(dirname "$evaluation")/evaluation.md)"
}

# flat_files DIR ARRAY: fills ARRAY with every file under DIR, sorted. The v1 arm's configs are
# one flat directory (the v1 checkout reads only configs/*.toml), so two files sharing a basename
# stop.
flat_files() {
    local -n flat_out="$2"
    local file seen=" "
    mapfile -d '' flat_out < <(find "$1" -type f -print0 | sort -z)
    for file in "${flat_out[@]}"; do
        [[ "$seen" != *" ${file##*/} "* ]] || stop "$1 holds two files named ${file##*/}"
        seen+="${file##*/} "
    done
}

# The v1 arm's configs: the freeze's, flattened, with the v1 commit's checks.toml; a rerun must find
# the same. Only checks.toml may differ; a file missing on either side stops.
v1_configs() {
    local arm="$1" ckv="$2" diff="$1/configs.diff" now="" file name
    local -a frozen=() entries=()
    if [[ ! -d "$arm/configs" ]]; then
        mkdir -p "$arm/configs.tmp/flat"
        git -C "$REPO_ROOT" archive HEAD configs | tar -x -C "$arm/configs.tmp"
        flat_files "$arm/configs.tmp/configs" frozen
        cp "${frozen[@]}" "$arm/configs.tmp/flat/"
        cp "$ckv/configs/checks.toml" "$arm/configs.tmp/flat/checks.toml"
        mv "$arm/configs.tmp/flat" "$arm/configs"
        rm -rf "$arm/configs.tmp"
    fi
    flat_files "$REPO_ROOT/configs" frozen
    for file in "${frozen[@]}"; do
        name="${file##*/}"
        [[ -f "$arm/configs/$name" ]] \
            || stop "the v1 arm's configs differ from the freeze's beyond checks.toml: no $name"
        if [[ "$name" == checks.toml ]]; then
            now="$(diff "$file" "$arm/configs/$name" || true)"
            [[ -z "$now" ]] || now="diff -r $file $arm/configs/$name"$'\n'"$now"
        elif ! cmp -s "$file" "$arm/configs/$name"; then
            stop "the v1 arm's configs differ from the freeze's beyond checks.toml: $name"
        fi
    done
    mapfile -d '' entries < <(find "$arm/configs" -mindepth 1 -print0)
    [[ "${#entries[@]}" == "${#frozen[@]}" ]] \
        || stop "the v1 arm's configs differ from the freeze's beyond checks.toml: extra files"
    if [[ -f "$diff" ]]; then
        [[ "$now" == "$(cat "$diff")" ]] || stop "$arm/configs changed since $diff"
    else
        printf '%s\n' "$now" >"$diff"
    fi
    note "v1 configs: the freeze's, flat, with $ckv/configs/checks.toml; configs.diff sha256" \
        "$(sha_of "$diff")"
}

cmd_v1arm() {
    hold "v1arm-$SET"
    completed look "$SET" || stop "look $SET has not completed: the v1 arm follows the look"
    local ckv="${PHYSVERIFY_FINAL_V1_CHECKOUT:-}" base="${PHYSVERIFY_FINAL_V1_BASE:-}"
    local overlay_cli="${PHYSVERIFY_FINAL_OVERLAY_CLI:-}" arm="${PHYSVERIFY_FINAL_V1_ARM:-$F/v1arm}"
    [[ -d "$ckv" ]] || stop "set PHYSVERIFY_FINAL_V1_CHECKOUT to the checkout at the v1 commit"
    [[ -f "$base/calibration.json" ]] || stop "set PHYSVERIFY_FINAL_V1_BASE to the base iteration"
    [[ -f "$overlay_cli" ]] || stop "set PHYSVERIFY_FINAL_OVERLAY_CLI to overlay_cli.py"
    ! near "$arm" "$CAL" || stop "the v1 arm never writes in $CAL"
    local v1 dirty state file meta evaluation
    v1="$(frozen v1.commit)"
    [[ "$(git -C "$ckv" rev-parse HEAD)" == "$v1" ]] || stop "$ckv is not at the v1 commit $v1"
    dirty="$(git -C "$ckv" status --porcelain -- src configs)" || stop "git status failed in $ckv"
    [[ -z "$dirty" ]] || stop "$ckv has uncommitted files: $dirty"
    [[ "$(sha_of "$base/calibration.json")" == "$(frozen v1.calibration_sha256)" ]] \
        || stop "$base/calibration.json is not the freeze's v1 calibration"
    mkdir -p "$arm/calibration"
    for file in calibration.json calibration/stage_errors.precise.json calibration/stage_sigma.json; do
        copy_once "$base/$file" "$arm/calibration/$(basename "$file")"
    done
    v1_configs "$arm" "$ckv"
    state="$(fq v1-state "$FREEZE" ${SPECS//,/ })" || stop "the v1 arm of $SPECS cannot run"
    note "$state of $SPECS at the v1 commit $v1, the base calibration $(sha_of "$arm/calibration/calibration.json")"
    run_logged 0 -- env -C "$ckv" PHYSVERIFY__PATHS__CALIBRATION_DIR="$arm/calibration" \
        PHYSVERIFY__PATHS__RUNS_DIR="$arm/runs" PHYSVERIFY__RUNTIME__HEAVY_LOCK="$arm/heavy.lock" \
        PHYSVERIFY_UNSEAL="$SPECS" PHYSVERIFY_UNSEAL_REPEAT=1 PYTHONPATH="$ckv/src" \
        "$PY" "$overlay_cli" "$arm/overlay" --config-dir "$arm/configs" --profile precise \
        --device cpu --set "runtime.threads=$THREADS" \
        calibrate evaluate --artifact "$arm/calibration/calibration.json" "${LOOK[@]}"
    meta="$(find "$arm/overlay" -name meta.json 2>/dev/null | grep -c . || true)"
    [[ "$meta" == 0 ]] || stop "the overlay holds $meta meta.json: the v1 arm recomputed stages"
    evaluation="$(evaluation_of "$LAST_LOG")"
    note "v1 evaluation $evaluation sha256 $(sha_of "$evaluation"); overlay meta.json 0;" \
        "report R4_jump and R4_size only"
}

check_frozen_doctor "$F" "$CAL" "$CACHE"
begin "$STEP" "${SET:-all}"
"cmd_$CMD"
row 0
echo "final: $SUMMARY"
