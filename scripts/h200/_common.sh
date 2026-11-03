# Shared helpers for the H200 scripts; sourced, never executed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LATEST_LINK="h200-timing-latest"
# The persistent per-user directory every H200 path derives from (see env.example.sh).
H200_ROOT="${PHYSVERIFY_H200_ROOT:-/home/mmonterot/work}"
H200_ENV="${PHYSVERIFY_H200_ENV:-$H200_ROOT/env.sh}"
# The CPU calibration chain's directory; the name is the existing on-disk layout.
CHAIN_ROOT="$H200_ROOT/a8v2"

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

# use_device NAME [FAIL]: the value of the variable NAME (cpu or cuda) becomes PV_DEVICE. cpu hides
# the GPUs; cuda needs one GPU of 0-3 in CUDA_VISIBLE_DEVICES (gpu_queue.sh; 4-7 are never used).
# FAIL (die by default) refuses anything else.
use_device() {
    local device="${!1:-}"
    case "$device" in
        cpu) export CUDA_VISIBLE_DEVICES="" ;;
        cuda)
            [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-3]$ ]] \
                || "${2:-die}" "$1=cuda needs CUDA_VISIBLE_DEVICES in 0-3 (gpu_queue.sh), got" \
                    "'${CUDA_VISIBLE_DEVICES-unset}'"
            ;;
        *) "${2:-die}" "$1 must be cpu or cuda, got '$device'" ;;
    esac
    PV_DEVICE="$device"
}

# require_pair WHAT DEVICE PROFILE [FAIL]: WHAT (an adopted calibration, a freeze) was made with
# this run's PV_DEVICE and PV_PROFILE; FAIL (die by default) refuses any other pair.
require_pair() {
    [[ "$2" == "$PV_DEVICE" && "$3" == "$PV_PROFILE" ]] && return 0
    "${4:-die}" "$1 is of device $2, profile $3, this run of device $PV_DEVICE, profile" \
        "$PV_PROFILE: calibration and evaluation never mix devices or profiles"
}

# A chain from before state.env recorded CHAIN_PROFILE ran this profile, the only one it had.
ASSUMED_CHAIN_PROFILE=precise

# state_value FILE KEY: KEY of the state.env FILE (empty when unset or when FILE is missing), read
# in a subshell so nothing here is overwritten.
state_value() {
    (
        # shellcheck disable=SC1090
        [[ -f "$1" ]] && source "$1"
        printf '%s' "${!2:-}"
    )
}

# require_chain_pair DIR [FAIL]: the calibration chain in DIR (its state.env) is of this run's pair
# (require_pair); a chain without CHAIN_PROFILE is taken as ASSUMED_CHAIN_PROFILE, and says so.
require_chain_pair() {
    local what="$1" profile
    profile="$(state_value "$1/state.env" CHAIN_PROFILE)"
    if [[ -z "$profile" ]]; then
        profile="$ASSUMED_CHAIN_PROFILE"
        what="$1 (no CHAIN_PROFILE: $profile assumed)"
    fi
    require_pair "$what" "$(state_value "$1/state.env" CHAIN_DEVICE)" "$profile" "${2:-die}"
}

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

# perception_arms.tsv: the declared arms of run_perception_iteration.sh, replayed by run_dev_b.sh.
ARMS_FILE="$(dirname "${BASH_SOURCE[0]}")/perception_arms.tsv"
CHECK_ID='R[0-9][a-z_]*'
SWAP_PATTERN="^(extraction\\.plan\\.[a-z]+=[a-z0-9_]+|check_variants\\.$CHECK_ID=[a-z0-9_]+"
SWAP_PATTERN+="|verification\\.thresholds\\.$CHECK_ID=[0-9]+(\\.[0-9]+)?"
SWAP_PATTERN+="|(verification\\.threshold_units|calibration\\.tau_tolerance_units)\\.$CHECK_ID=[a-z0-9_]+)\$"

# arm_swaps NAME: the swaps of a declared arm, one per line (exit 2 if undeclared).
arm_swaps() {
    local lines swap
    [[ "$1" =~ ^[a-z0-9][a-z0-9-]*$ ]] || die "the arm must be an arm name, got '$1'"
    [[ -f "$ARMS_FILE" ]] || die "no arm table at $ARMS_FILE"
    lines="$(awk -F '\t' -v arm="$1" '$1 == arm { print $2 }' "$ARMS_FILE")"
    [[ -n "$lines" && "$lines" != *$'\n'* ]] \
        || die "arm $1 is not declared exactly once in $ARMS_FILE"
    for swap in $lines; do
        [[ "$swap" =~ $SWAP_PATTERN ]] \
            || die "arm $1: '$swap' is not extraction.plan.<stage>=<provider>," \
                "check_variants.<check>=<variant> or a check's threshold, threshold unit or" \
                "tau tolerance unit"
        printf '%s\n' "$swap"
    done
}
