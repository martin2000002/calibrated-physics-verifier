#!/usr/bin/env bash
# The fine-tuning study after the calibration freeze, as one resumable chain on the H200:
#   rescore train and dev -> pairs -> stop rule -> beta grid + Flow-DPO LoRA -> checkpoint
#   selection -> AutoEval gate and dev judge -> pre-registration -> eval generation (base K = 8,
#   ft K = 4) -> eval scores -> eval judge (gate passed only) -> compare -> demo scenes.
#
#   run_finetune_chain.sh all            every step from the first not completed (a rerun resumes)
#   run_finetune_chain.sh STEP           that step alone, after every earlier one completed (list: steps)
#   run_finetune_chain.sh review ARGS    finetune pilot-review ARGS (--sample, or --better N [--note TEXT])
#   run_finetune_chain.sh status | steps
#
# Each step's last exit is in $D/steps.tsv; a step with exit 0 is skipped by `all`, and every
# physverify call resumes on its own. GPU steps run one by one through the GPU slot script, under
# nice; the others hide the GPUs (the cache keys keep the cuda device). Exit 3: the stop rule waits
# only for the hand review (run `review --sample`, then `review --better N`, then `all`). Exit 4:
# the stop rule fired for another reason, so the generator comparison is not evaluable with this
# verifier (the negative result; nothing of eval is scored).
#
# Required:
#   PHYSVERIFY_FT_CALIBRATION   the frozen calibration directory (paths.calibration_dir)
# Optional (defaults in brackets):
#   PHYSVERIFY_FT_DIR           the run's bookkeeping [$H200_ROOT/finetune-chain]
#   PHYSVERIFY_FT_DATA          paths.data_dir [$H200_ROOT/Repo/data]
#   PHYSVERIFY_FT_PYTHON        interpreter [/home/mmonterot/dev/venv-refactor/bin/python]
#   PHYSVERIFY_FT_GPU_SLOT      GPU slot command, called as SLOT TAG CMD... [$H200_ROOT/scratch/gpu_slot.sh];
#                               empty: run in place with CUDA_VISIBLE_DEVICES already in 0-3
#   PHYSVERIFY_FT_SET           extra KEY=VALUE overrides of every call, space separated (the
#                               scorer or pair rule: finetune.scorer=..., finetune.pairs....)
#   PHYSVERIFY_FT_STOP_RULE_DEVIATION   declared deviation passed to the commands the stop rule gates
#   PHYSVERIFY_FT_THREADS       runtime.threads [16]
# A rerun of the same $D must keep the calibration, the overrides, the deviation and the commit.
set -euo pipefail
SELF="$(realpath "${BASH_SOURCE[0]}")"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/_steps.sh"
cd "$REPO_ROOT"

USAGE="usage: run_finetune_chain.sh all | STEP | review ARGS | status | steps"
STEP_NAMES=(score-train score-dev pairs beta-grid train select gate-autoeval judge-dev freeze
    generate-base generate-ft score-base score-ft judge-eval compare demo)
GPU_STEPS=" score-train score-dev beta-grid train select gate-autoeval judge-dev generate-base"
GPU_STEPS+=" generate-ft score-base score-ft judge-eval demo "
# Steps whose failure leaves the rest valid: a failed gate means a verifier-only verdict.
OPTIONAL_STEPS=" gate-autoeval judge-dev judge-eval demo "
# The freeze records the gate and the dev judge: neither may run after it.
BEFORE_FREEZE=" gate-autoeval judge-dev "
WAITING_REVIEW=3
NOT_EVALUABLE=4
# Every stop-rule reason the hand review can clear holds this text (finetune/stop_rule.py).
REVIEW_REASON="pilot review"
# A stale stamp is a state to rebuild, not a verdict: the beta grid keeps its exit 2.
STALE_REASON="another scorer stamp"

# step_index NAME: its 1-based number (exit 2 when unknown).
step_index() {
    local k
    for k in "${!STEP_NAMES[@]}"; do
        [[ "${STEP_NAMES[$k]}" == "$1" ]] && { echo $((k + 1)); return 0; }
    done
    die "unknown step '$1'; steps: ${STEP_NAMES[*]}"
}

is_gpu_step() {
    [[ "$GPU_STEPS" == *" $1 "* ]]
}

is_optional() {
    [[ "$OPTIONAL_STEPS" == *" $1 "* ]]
}

deviation_args() {
    [[ -z "${PHYSVERIFY_FT_STOP_RULE_DEVIATION:-}" ]] \
        || printf '%s\0' --stop-rule-deviation "$PHYSVERIFY_FT_STOP_RULE_DEVIATION"
}

gate_passed() {
    gate_passed_in "$(pv compare-generators status)"
}

# body NAME: the physverify calls of one step (inside the GPU slot for a GPU step).
body() {
    local deviation=()
    mapfile -d '' -t deviation < <(deviation_args)
    case "$1" in
        score-train) pv finetune score --split train ;;
        score-dev) pv finetune score --split dev ;;
        pairs) pv finetune pairs && pv finetune status ;;
        beta-grid) pv finetune beta-grid "${deviation[@]}" ;;
        train) pv finetune train --resume "${deviation[@]}" ;;
        select) pv finetune select ;;
        gate-autoeval) pv compare-generators gate-autoeval ;;
        judge-dev)
            gate_passed || { echo "the AutoEval gate has not passed: no dev judge rows" >&2; return 1; }
            pv compare-generators judge --split dev
            ;;
        freeze) pv compare-generators freeze "${deviation[@]}" ;;
        generate-base) pv compare-generators generate --arm base "${deviation[@]}" ;;
        generate-ft) pv compare-generators generate --arm ft "${deviation[@]}" ;;
        score-base) pv compare-generators score --arm base ;;
        score-ft) pv compare-generators score --arm ft ;;
        judge-eval)
            gate_passed || { echo "the AutoEval gate has not passed: verifier-only verdict" >&2; return 1; }
            pv compare-generators judge --split eval
            ;;
        compare) pv compare-generators compare ;;
        demo) pv demo scenes --from-select --out "$D/demo" ;;
    esac
}

# stop_rule_exit LOG: WAITING_REVIEW when every reason of the fired rule in LOG is the hand
# review's, NOT_EVALUABLE otherwise.
stop_rule_exit() {
    local reasons
    reasons="$(sed -n 's/.*pilot stop rule fired: \(.*\); the generator comparison is not .*/\1/p' "$1" | tail -n 1)"
    [[ -n "$reasons" && "$reasons" != *"$STALE_REASON"* ]] || return 1
    if awk -v RS='; ' -v marker="$REVIEW_REASON" 'NF && index($0, marker) == 0 { bad = 1 } END { exit bad }' \
        <<<"$reasons"; then
        echo "$WAITING_REVIEW"
    else
        echo "$NOT_EVALUABLE"
    fi
}

run_step() {
    local name="$1" n code=0 stop_code
    n="$(step_index "$name")"
    begin "$n" "$name"
    CALL_K=1
    LAST_LOG="$LOGS/$n-$name.log"
    set +e
    if is_gpu_step "$name" && [[ -n "$SLOT" ]]; then
        "$SLOT" "ft-$name" bash "$SELF" --body "$name" >"$LAST_LOG" 2>&1
    elif is_gpu_step "$name"; then
        use_device PV_DEVICE
        body "$name" >"$LAST_LOG" 2>&1
    else
        CUDA_VISIBLE_DEVICES="" body "$name" >"$LAST_LOG" 2>&1
    fi
    code=$?
    set -e
    tail -n 5 "$LAST_LOG"
    if [[ "$name" == beta-grid && "$code" -eq 2 ]] && stop_code="$(stop_rule_exit "$LAST_LOG")"; then
        row "$stop_code"
        if [[ "$stop_code" == "$WAITING_REVIEW" ]]; then
            note "stop rule waits for the hand review: run_finetune_chain.sh review --sample, then --better N"
        else
            note "stop rule fired: the generator comparison is not evaluable with this verifier (negative result), see $LAST_LOG"
        fi
        exit "$stop_code"
    fi
    row "$code"
    if [[ "$code" -ne 0 && "$code" -ne "$LOCK_BUSY" ]] && is_optional "$name"; then
        note "step $n ($name) exited $code; the chain goes on without it, see $LAST_LOG"
        return 0
    fi
    [[ "$code" -eq 0 ]] || die "step $n ($name) failed ($code); log $LAST_LOG"
}

completed() {
    [[ "$(last_exit "$1")" == 0 ]]
}

# done_step N: completed, or an optional step that ran and failed other than on a busy lock
# (never rerun by `all`: a dev judge after the freeze would change what the freeze recorded).
done_step() {
    local code
    code="$(last_exit "$1")"
    [[ "$code" == 0 ]] && return 0
    is_optional "${STEP_NAMES[$(($1 - 1))]}" && [[ "$code" =~ ^[0-9]+$ && "$code" != "$LOCK_BUSY" ]]
}

# require_before N: every earlier step done, and nothing the freeze records asked after it.
require_before() {
    local k name="${STEP_NAMES[$(($1 - 1))]}"
    for ((k = 1; k < $1; k++)); do
        done_step "$k" || die "step $k (${STEP_NAMES[$((k - 1))]}) has not completed: run 'all'"
    done
    if [[ "$BEFORE_FREEZE" == *" $name "* && -n "$(last_exit "$(step_index freeze)")" ]]; then
        die "step $1 ($name) cannot run after the freeze, which recorded its outcome"
    fi
}

# check_fingerprint: a rerun of $D keeps what its earlier steps ran under.
check_fingerprint() {
    local key value
    for key in FT_CALIBRATION FT_SET FT_DEVIATION FT_COMMIT; do
        if ! grep -q "^$key=" "$STATE" 2>/dev/null; then
            save "$key" "${!key}"
            continue
        fi
        value="$(state_value "$STATE" "$key")"
        if [[ "$value" != "${!key}" ]]; then
            die "$D ran with $key='$value', not '${!key}': use another PHYSVERIFY_FT_DIR"
        fi
    done
}

setup() {
    source_h200_env
    [[ ! -e "$REPO_ROOT/configs/local.toml" ]] \
        || die "configs/local.toml would override the H200 paths: remove it from this checkout"
    [[ -n "${PHYSVERIFY_FT_CALIBRATION:-}" && -d "$PHYSVERIFY_FT_CALIBRATION" ]] \
        || die "set PHYSVERIFY_FT_CALIBRATION to the frozen calibration directory"
    D="${PHYSVERIFY_FT_DIR:-$H200_ROOT/finetune-chain}"
    DATA="${PHYSVERIFY_FT_DATA:-$H200_ROOT/Repo/data}"
    PY="${PHYSVERIFY_FT_PYTHON:-/home/mmonterot/dev/venv-refactor/bin/python}"
    SLOT="${PHYSVERIFY_FT_GPU_SLOT-$H200_ROOT/scratch/gpu_slot.sh}"
    THREADS="${PHYSVERIFY_FT_THREADS:-16}"
    PV_RUNNER=py
    PV_NICE=(nice -n 10)
    PV_EXTRA=()
    local override overrides=()
    read -ra overrides <<<"${PHYSVERIFY_FT_SET:-}"
    for override in "${overrides[@]}"; do
        [[ "$override" == *=* ]] || die "PHYSVERIFY_FT_SET entries are KEY=VALUE, got '$override'"
        PV_EXTRA+=(--set "$override")
    done
    export PHYSVERIFY__PATHS__DATA_DIR="$DATA"
    export PHYSVERIFY__PATHS__CALIBRATION_DIR="$PHYSVERIFY_FT_CALIBRATION"
    export PHYSVERIFY__PATHS__RUNS_DIR="$D/runs"
    export PHYSVERIFY__PATHS__WEIGHTS_DIR="$DATA/weights"
    export PHYSVERIFY__PATHS__VENDOR_DIR="$DATA/vendor"
    export PHYSVERIFY__PATHS__HF_HOME="${HF_HOME:-$H200_ROOT/.cache/huggingface}"
    export PHYSVERIFY__RUNTIME__HEAVY_LOCK="$D/heavy.lock"
    export PHYSVERIFY__H3__AUTOEVAL__ROOT="${PHYSVERIFY__H3__AUTOEVAL__ROOT:-$H200_ROOT}"
    LOGS="$D/logs" STATE="$D/state.env" STEPS="$D/steps.tsv" NOTES="$D/notes.md"
    SUMMARY="$D/summary.md" SUMMARY_TITLE="Fine-tuning chain after the freeze"
    SUMMARY_INFO="calibration $PHYSVERIFY_FT_CALIBRATION; overrides '${PHYSVERIFY_FT_SET:-}'"
    mkdir -p "$LOGS" "$D/runs"
    touch "$STEPS" "$NOTES"
    FT_CALIBRATION="$(realpath "$PHYSVERIFY_FT_CALIBRATION")"
    FT_SET="${PHYSVERIFY_FT_SET:-}"
    FT_DEVIATION="${PHYSVERIFY_FT_STOP_RULE_DEVIATION:-}"
    FT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
}

[[ $# -ge 1 ]] || die "$USAGE"
case "$1" in
    steps)
        printf '%s\n' "${STEP_NAMES[@]}"
        ;;
    --body)
        [[ $# -eq 2 ]] || die "$USAGE"
        setup
        use_device PV_DEVICE
        step_index "$2" >/dev/null
        body "$2"
        ;;
    review)
        setup
        check_fingerprint
        CUDA_VISIBLE_DEVICES="" pv finetune pilot-review "${@:2}"
        ;;
    status)
        setup
        export CUDA_VISIBLE_DEVICES=""
        cat "$STEPS"
        pv finetune status
        pv compare-generators status
        ;;
    all)
        [[ $# -eq 1 ]] || die "$USAGE"
        setup
        check_fingerprint
        for k in "${!STEP_NAMES[@]}"; do
            done_step $((k + 1)) || run_step "${STEP_NAMES[$k]}"
        done
        ;;
    *)
        [[ $# -eq 1 ]] || die "$USAGE"
        n="$(step_index "$1")"
        setup
        require_before "$n"
        check_fingerprint
        run_step "$1"
        ;;
esac
