#!/usr/bin/env bash
# H3 (ADR 0013): fine-tuned vs base vs best-of-8. Every step resumes; run them in this order:
#   run_h3.sh base-eval      base eval videos (outcome-blind: may run before anything else)
#   run_h3.sh dev            AutoEval on the dev videos (after the gate and `finetune select`)
#   run_h3.sh freeze         the pre-registration (before any eval score or judge row)
#   run_h3.sh eval           fine-tuned eval videos, scores, eval judge rows, compare (without a
#                            passed AutoEval gate the judge is skipped: verifier-only verdict)
#   run_h3.sh compare | status
# Extra arguments go to the physverify subcommand (for example --shard 1/2 or --limit 4).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"
source_h200_env

# The `h3 status` line of a passed gate (physverify.cli.commands.h3 GATE_LINE + GATE_PASSED).
GATE_PASSED_LINE="  gate: passed"
USAGE="usage: run_h3.sh base-eval | dev | freeze | eval | compare | status [ARGS...]"
export PHYSVERIFY__H3__AUTOEVAL__ROOT="${PHYSVERIFY__H3__AUTOEVAL__ROOT:-$H200_ROOT}"

[[ $# -ge 1 ]] || die "$USAGE"
step="$1"
shift

case "$step" in
    base-eval)
        require_cuda
        pv h3 generate --arm base "$@"
        ;;
    dev)
        require_cuda
        pv h3 judge --split dev "$@"
        ;;
    freeze)
        pv h3 freeze "$@"
        ;;
    eval)
        [[ $# -eq 0 ]] || die "$USAGE"
        require_cuda
        pv h3 generate --arm ft
        pv h3 score --arm base
        pv h3 score --arm ft
        h3_status="$(pv h3 status)"
        if grep -qx "$GATE_PASSED_LINE" <<<"$h3_status"; then
            pv h3 judge --split eval
        else
            printf '%s\n' "the AutoEval gate has not passed: no eval judge rows" >&2
        fi
        pv h3 compare
        ;;
    compare)
        pv h3 compare "$@"
        ;;
    status)
        pv h3 status "$@"
        ;;
    *)
        die "$USAGE"
        ;;
esac
