#!/usr/bin/env bash
# H3 (ADR 0013): fine-tuned vs base vs best-of-8. Every step resumes; run them in this order:
#   run_generator_comparison.sh base-eval      base eval videos (outcome-blind: may run before anything else)
#   run_generator_comparison.sh dev            AutoEval on the dev videos (after the gate and `finetune select`)
#   run_generator_comparison.sh freeze         the pre-registration (before any eval score or judge row)
#   run_generator_comparison.sh eval           fine-tuned eval videos, scores, eval judge rows, compare (without a
#                            passed AutoEval gate the judge is skipped: verifier-only verdict)
#   run_generator_comparison.sh compare | status
# Extra arguments go to the physverify subcommand (for example --shard 1/2 or --limit 4).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"
source_h200_env

# The `compare-generators status` line of a passed gate (physverify.generator_comparison.status GATE_LINE + GATE_PASSED).
GATE_PASSED_LINE="  gate: passed"
USAGE="usage: run_generator_comparison.sh base-eval | dev | freeze | eval | compare | status [ARGS...]"
export PHYSVERIFY__H3__AUTOEVAL__ROOT="${PHYSVERIFY__H3__AUTOEVAL__ROOT:-$H200_ROOT}"

[[ $# -ge 1 ]] || die "$USAGE"
step="$1"
shift

case "$step" in
    base-eval)
        require_cuda
        pv compare-generators generate --arm base "$@"
        ;;
    dev)
        require_cuda
        pv compare-generators judge --split dev "$@"
        ;;
    freeze)
        pv compare-generators freeze "$@"
        ;;
    eval)
        [[ $# -eq 0 ]] || die "$USAGE"
        require_cuda
        pv compare-generators generate --arm ft
        pv compare-generators score --arm base
        pv compare-generators score --arm ft
        h3_status="$(pv compare-generators status)"
        if grep -qx "$GATE_PASSED_LINE" <<<"$h3_status"; then
            pv compare-generators judge --split eval
        else
            printf '%s\n' "the AutoEval gate has not passed: no eval judge rows" >&2
        fi
        pv compare-generators compare
        ;;
    compare)
        pv compare-generators compare "$@"
        ;;
    status)
        pv compare-generators status "$@"
        ;;
    *)
        die "$USAGE"
        ;;
esac
