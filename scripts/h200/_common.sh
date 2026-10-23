# Shared helpers for the H200 scripts; sourced, never executed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LATEST_LINK="h200-timing-latest"
# The persistent per-user directory every H200 path derives from (see env.example.sh).
H200_ROOT="${PHYSVERIFY_H200_ROOT:-/home/mmonterot/work}"
H200_ENV="${PHYSVERIFY_H200_ENV:-$H200_ROOT/env.sh}"

die() {
    printf '%s\n' "$*" >&2
    exit 2
}

has_cuda() {
    local gpus
    gpus="$(nvidia-smi -L 2>/dev/null || true)"
    grep -q '^GPU ' <<<"$gpus"
}

# Checked before anything touches uv: without a CUDA GPU nothing here can measure.
require_cuda() {
    command -v nvidia-smi >/dev/null 2>&1 \
        || die "no CUDA GPU (nvidia-smi not found); this benchmark only runs on the university H200"
    has_cuda \
        || die "no CUDA GPU (nvidia-smi lists none); this benchmark only runs on the university H200"
}

source_h200_env() {
    [[ -f "$H200_ENV" ]] || return 0
    set +u
    # shellcheck disable=SC1090
    source "$H200_ENV"
    set -u
}

# Prints the pinned Kubric image (runtime.docker_image) after pulling it when Docker lacks it;
# $1 is the physverify wrapper whose configuration applies.
ensure_kubric_image() {
    local image
    image="$("$1" doctor 2>/dev/null | sed -n 's/^docker_image: //p' || true)"
    if [[ -n "$image" && "$image" != "(unset)" ]] && ! docker image inspect "$image" >/dev/null 2>&1; then
        "$1" synth pull-image >&2 || return 1
    fi
    printf '%s\n' "$image"
}

# runtime.threads is set explicitly so the run records the CPU budget it had (default: nproc).
THREADS="${PHYSVERIFY_TIMING_THREADS:-$(nproc)}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] \
    || die "PHYSVERIFY_TIMING_THREADS must be a positive integer, got '$THREADS'"

# A script that may run without a GPU (run_demo.sh scoring a given video) sets PV_DEVICE=cpu, before
# or after sourcing this file; an exported PV_DEVICE is kept.
PV_DEVICE="${PV_DEVICE:-cuda}"

# PV_RUNNER picks how pv reaches the checkout: uv (the default) or py, the interpreter $PY with
# this checkout's src first (_steps.sh). PV_PROFILE is precise; PV_NICE prefixes and PV_EXTRA
# extends every call. A caller changes them after sourcing; exported values are ignored.
PV_RUNNER=uv
PV_PROFILE=precise
PV_NICE=()
PV_EXTRA=()

pv() {
    local runner=(uv run --no-sync physverify)
    if [[ "$PV_RUNNER" == py ]]; then
        runner=(env PYTHONPATH="$REPO_ROOT/src" "$PY" -c "from physverify.cli.main import main; main()")
    fi
    "${PV_NICE[@]}" "${runner[@]}" --profile "$PV_PROFILE" --device "$PV_DEVICE" \
        --set "runtime.threads=$THREADS" "${PV_EXTRA[@]}" "$@"
}

# h200_py SCRIPT ARGS: scripts/h200/SCRIPT of this checkout through $PY.
h200_py() {
    env PYTHONPATH="$REPO_ROOT/src" "$PY" "$REPO_ROOT/scripts/h200/$1" "${@:2}"
}

# query ARGS: run_query.py of this checkout.
query() {
    h200_py run_query.py "$@"
}
