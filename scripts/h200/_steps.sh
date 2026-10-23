# Step bookkeeping of the H200 runners (steps.tsv, state.env, notes, summary, logged calls);
# sourced after _common.sh, never executed.
#
# The caller sets LOGS, STATE, STEPS, NOTES and SUMMARY, the summary's SUMMARY_TITLE
# and SUMMARY_INFO line, and PY, PV_DEVICE and THREADS for pv (DOCTOR before doctor_value).
# _common.sh sets PV_PROFILE to precise and ignores an exported one (run_human_agreement.sh always
# runs precise); a caller changes it after sourcing. require_completed reads STEP_SAVES (step ->
# keys). The runners call the checkout through the interpreter $PY, never through uv.

PV_RUNNER=py

sha_of() {
    sha256sum "$1" | cut -d ' ' -f 1
}

# record_sha_of LOG: the decision record's sha256 that `exclusions show` printed in LOG (the hash
# check_scope and the human-agreement pre-registration carry).
record_sha_of() {
    sed -n 's/^record: .* sha256 \([0-9a-f]*\)$/\1/p' "$1"
}

# fq ARGS: scripts/h200/freeze_query.py of this checkout (the freeze and the final looks).
fq() {
    h200_py freeze_query.py "$@"
}

# copy_once FILE TARGET: FILE copied to TARGET unless TARGET already holds it; a different TARGET
# stops the step (a frozen directory adopts its artifacts once and never mixes them).
copy_once() {
    if [[ -f "$2" ]]; then
        cmp -s "$1" "$2" || stop "$2 differs from $1: a frozen artifact is never replaced"
        return
    fi
    cp "$1" "$2.tmp"
    mv "$2.tmp" "$2"
}

doctor_value() {
    sed -n "s/^$1: //p" <<<"$DOCTOR" | sed 's/ (.*)$//'
}

same_path() {
    [[ -n "$1" && "$(realpath -m "$1")" == "$(realpath -m "$2")" ]]
}

save() {
    printf -v "$1" '%s' "$2"
    printf '%s=%q\n' "$1" "$2" >>"$STATE"
}

render_summary() {
    {
        printf '# %s\n\n%s\n\n## Notes\n\n' "$SUMMARY_TITLE" "$SUMMARY_INFO"
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

# last_exit N [STEPS_FILE]: the exit code of step N's last row (empty when it has none).
last_exit() {
    local file="${2:-$STEPS}"
    [[ -f "$file" ]] || return 0
    awk -F '\t' -v k="$1" '$1 == k { c = $3 } END { print c }' "$file"
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

# require_completed FIRST FROM: steps FIRST..FROM-1 last recorded with exit 0 and with their
# STEP_SAVES keys set (a tolerated exit 1 is recorded as 0).
require_completed() {
    local k code key
    for ((k = $1; k < $2; k++)); do
        code="$(last_exit "$k")"
        [[ "$code" == 0 ]] || die "step $k has not completed (last exit '${code:-none}'): rerun from $k"
        for key in ${STEP_SAVES[$k]:-}; do
            [[ -n "${!key:-}" ]] || die "step $k recorded no $key in $STATE: rerun from $k"
        done
    done
}

call() {
    local tolerated="$1"
    shift 2
    run_logged "$tolerated" -- pv "$@"
}

# evaluation_of LOG: the evaluation.json calibrate evaluate printed in LOG; a skipped real dataset
# stops the step (a look judges every set it names).
evaluation_of() {
    local evaluation
    if grep -q '^real dataset .* skipped' "$1"; then
        stop "a real dataset was skipped (see $1)"
    fi
    evaluation="$(sed -n '/^evaluation: /{s///p;q}' "$1")"
    [[ -n "$evaluation" && -f "$evaluation" ]] || stop "no evaluation in $1"
    printf '%s\n' "$evaluation"
}

# The run directory the last call printed.
run_of() {
    local run_dir
    run_dir="$(sed -n '/^run: /{s///p;q}' "$LAST_LOG")"
    [[ -n "$run_dir" && -d "$run_dir" ]] || stop "no run directory in $LAST_LOG"
    printf '%s\n' "$run_dir"
}
