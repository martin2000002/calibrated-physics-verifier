#!/usr/bin/env bash
# End-to-end demo (ADR 0013, D1): verifier reports and overlay videos.
#   run_demo.sh scenes                      base vs fine-tuned fixed scenes (after `compare-generators freeze`)
#   run_demo.sh run --prompt TEXT --query PHRASE [...] [--adapter DIR]   generate, then verify
#   run_demo.sh run --prompt TEXT --query PHRASE --video FILE            verify a given video
#                                                       (on CPU when the machine has no GPU)
# Output goes to $PHYSVERIFY_DEMO_OUT (default $PHYSVERIFY_H200_ROOT/demo).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"
source_h200_env

USAGE="usage: run_demo.sh scenes | run --prompt TEXT --query PHRASE [--video FILE | --adapter DIR]"
OUT="${PHYSVERIFY_DEMO_OUT:-$H200_ROOT/demo}"

[[ $# -ge 1 ]] || die "$USAGE"
mode="$1"
shift

has_video() {
    local arg
    for arg in "$@"; do
        [[ "$arg" == --video || "$arg" == --video=* ]] && return 0
    done
    return 1
}

case "$mode" in
    scenes)
        require_cuda
        pv demo scenes --from-select --out "$OUT" "$@"
        ;;
    run)
        if has_video "$@" && ! has_cuda; then
            PV_DEVICE="cpu"
        else
            require_cuda
        fi
        pv demo run --out "$OUT" "$@"
        ;;
    *)
        die "$USAGE"
        ;;
esac
