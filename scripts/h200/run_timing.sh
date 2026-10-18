#!/usr/bin/env bash
# The single H200 command: setup, render prerequisites, one timing run, latest-run link.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"

# The Kubric scenes kubric_render times. Not fatal: without them that target is skipped with its
# reason and the other targets are still measured.
prepare_render() {
    if ! docker info >/dev/null 2>&1; then
        echo "docker does not respond: kubric_render will be skipped" >&2
        return 0
    fi
    ensure_kubric_image pv >/dev/null || return 1
    pv dataset generate --subset smoke \
        && pv dataset simulate --subset smoke \
        && pv dataset perturb --subset smoke
}

require_cuda
bash scripts/h200/setup.sh
pv timing env
prepare_render || echo "render prerequisites failed: kubric_render may be skipped" >&2

log="$(mktemp "${TMPDIR:-/tmp}/h200-timing.XXXXXX.log")"
set +e
pv timing run --dataset local_video --subset timing_wan21 | tee "$log"
code="${PIPESTATUS[0]}"
set -e

run_dir="$(sed -n '/^run: /{s///p;q}' "$log")"
if [[ -n "$run_dir" && -d "$run_dir" ]]; then
    mv "$log" "$run_dir.log"
    ln -sfn "$(basename "$run_dir")" "$(dirname "$run_dir")/$LATEST_LINK"
    echo "latest run: $(dirname "$run_dir")/$LATEST_LINK (summary: timing/summary.md, log: $run_dir.log)"
else
    echo "timing run created no run directory; its output is in $log" >&2
fi
exit "$code"
