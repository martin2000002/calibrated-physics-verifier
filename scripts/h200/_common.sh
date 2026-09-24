# Shared helpers for the H200 scripts; sourced, never executed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LATEST_LINK="h200-timing-latest"

die() {
    printf '%s\n' "$*" >&2
    exit 2
}

# Checked before anything touches uv: without a CUDA GPU nothing here can measure.
require_cuda() {
    command -v nvidia-smi >/dev/null 2>&1 \
        || die "no CUDA GPU (nvidia-smi not found); this benchmark only runs on the university H200"
    local gpus
    gpus="$(nvidia-smi -L 2>/dev/null || true)"
    grep -q '^GPU ' <<<"$gpus" \
        || die "no CUDA GPU (nvidia-smi lists none); this benchmark only runs on the university H200"
}

# runtime.threads is set explicitly so the run records the CPU budget it had (default: nproc).
THREADS="${PHYSVERIFY_TIMING_THREADS:-$(nproc)}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] \
    || die "PHYSVERIFY_TIMING_THREADS must be a positive integer, got '$THREADS'"

pv() {
    uv run --no-sync physverify --profile precise --device cuda --set "runtime.threads=$THREADS" "$@"
}
