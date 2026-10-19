#!/usr/bin/env bash
# Builds the Kubric synthetic dataset on the H200 (CPU only) through Docker or udocker.
# Usage: bash scripts/h200/run_kubric.sh [subset]   (default: full; resumes when run again)
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
source_h200_env

SUBSET="${1:-full}"
ENGINE="${PHYSVERIFY_KUBRIC_ENGINE:-docker}"
PROFILE="${PHYSVERIFY_KUBRIC_PROFILE:-precise}"
KUBRIC_THREADS="${PHYSVERIFY_KUBRIC_THREADS:-16}"
WORKERS="${PHYSVERIFY_KUBRIC_WORKERS:-12}"
JOBS="${PHYSVERIFY_KUBRIC_JOBS:-8}"
MIN_FREE_GB="${PHYSVERIFY_KUBRIC_MIN_FREE_GB:-50}"
UDOCKER="${PHYSVERIFY_UDOCKER:-$H200_ROOT/udocker-venv/bin/udocker}"
CONTAINER="${PHYSVERIFY_UDOCKER_CONTAINER:-kub}"
DATA_DIR="${PHYSVERIFY__PATHS__DATA_DIR:-$H200_ROOT/kubric-data}"
IMAGE_USED=""

for name in KUBRIC_THREADS WORKERS JOBS MIN_FREE_GB; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got '${!name}'"
done
[[ "$ENGINE" == docker || "$ENGINE" == udocker ]] \
    || die "PHYSVERIFY_KUBRIC_ENGINE must be docker or udocker, got '$ENGINE'"

# Datadog's ddtrace in the host Python breaks udocker unless it is switched off.
export DD_TRACE_ENABLED=false DD_SYMBOL_DATABASE_UPLOAD_ENABLED=false
export DD_INSTRUMENTATION_TELEMETRY_ENABLED=false DD_REMOTE_CONFIGURATION_ENABLED=false
export PHYSVERIFY__CONTAINER__ENGINE="$ENGINE"
if [[ "$ENGINE" == udocker ]]; then
    export PHYSVERIFY__CONTAINER__UDOCKER_BIN="$UDOCKER"
    export PHYSVERIFY__CONTAINER__UDOCKER_CONTAINER="$CONTAINER"
fi
export PHYSVERIFY__PATHS__DATA_DIR="$DATA_DIR"
# A Kubric-only lock: a second build exits 75 at once and GPU jobs are not queued behind it.
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="${PHYSVERIFY_KUBRIC_LOCK:-$H200_ROOT/kubric.lock}"
export PHYSVERIFY__RUNTIME__LOCK_TIMEOUT_S=0

kv() {
    local profile_option
    mapfile -t profile_option < <(error_profile_option "$PROFILE")
    uv run --no-sync physverify --profile "$PROFILE" "${profile_option[@]}" \
        --set "runtime.threads=$KUBRIC_THREADS" \
        --set "dataset.render.workers=$WORKERS" \
        --set "dataset.render.jobs_per_process=$JOBS" "$@"
}

prepare_docker() {
    docker info >/dev/null 2>&1 || die "docker does not respond (is the user in the docker group?)"
    IMAGE_USED="$(ensure_kubric_image kv)" || die "pulling the pinned Kubric image failed"
    [[ -n "$IMAGE_USED" && "$IMAGE_USED" != "(unset)" ]] || die "runtime.docker_image is not set"
}

# The image column of `udocker ps` for the named container (empty when it is not listed).
udocker_image_of() {
    local line image
    line="$("$UDOCKER" ps 2>/dev/null | grep -m 1 -F "'$CONTAINER'")" || return 0
    read -r image <<<"${line##*]}"
    printf '%s\n' "$image"
}

# The container must come from runtime.docker_image (the pinned digest), or from
# PHYSVERIFY_KUBRIC_IMAGE when that is set explicitly.
prepare_udocker() {
    [[ -x "$UDOCKER" ]] || die "udocker not found at $UDOCKER (set PHYSVERIFY_UDOCKER)"
    local expected="${PHYSVERIFY_KUBRIC_IMAGE:-$(sed -n 's/^docker_image: //p' <<<"$DOCTOR")}"
    [[ -n "$expected" && "$expected" != "(unset)" ]] \
        || die "runtime.docker_image is not set (or set PHYSVERIFY_KUBRIC_IMAGE)"
    if ! "$UDOCKER" inspect "$CONTAINER" >/dev/null 2>&1; then
        "$UDOCKER" pull "$expected" || die "udocker pull $expected failed"
        "$UDOCKER" create --name="$CONTAINER" "$expected" \
            || die "udocker create $CONTAINER failed"
    fi
    local remedy="remove it with '$UDOCKER rm $CONTAINER' and run this script again to create it"
    IMAGE_USED="$(udocker_image_of)"
    [[ -n "$IMAGE_USED" ]] \
        || die "cannot verify the image of the udocker container $CONTAINER ('udocker ps' does" \
            "not list it); $remedy from $expected"
    [[ "$IMAGE_USED" == "$expected" ]] \
        || die "the udocker container $CONTAINER comes from $IMAGE_USED, not $expected; $remedy," \
            "or set PHYSVERIFY_KUBRIC_IMAGE=$IMAGE_USED to use it knowingly"
}

mkdir -p "$DATA_DIR/runs"
FREE_GB="$(df --output=avail -B1G "$DATA_DIR" | tail -n 1 | tr -d ' ')" || die "df failed on $DATA_DIR"
[[ "$FREE_GB" =~ ^[0-9]+$ ]] || die "could not read the free space of $DATA_DIR"
((FREE_GB >= MIN_FREE_GB)) || die "only ${FREE_GB} GB free on $DATA_DIR, need $MIN_FREE_GB"

cd "$REPO_ROOT"
DOCTOR="$(kv doctor)" || die "physverify doctor failed"
grep -q "^container_engine: $ENGINE\b" <<<"$DOCTOR" \
    || die "physverify doctor does not report container_engine: $ENGINE"
"prepare_$ENGINE"
# The same runner, mounts and workdir as the build, importing the worker from the mounted source.
kv doctor --probe-container >&2 \
    || die "the $ENGINE container cannot import kubric, bpy, pybullet and the Kubric worker"

LOG="$DATA_DIR/runs/kubric-$SUBSET-$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$(basename "$LOG")" "$DATA_DIR/runs/kubric-latest.log"
printf 'container_engine: %s\ncontainer_image: %s\n' "$ENGINE" "$IMAGE_USED" >"$LOG"
rc=0
kv dataset build --subset "$SUBSET" 2>&1 | tee -a "$LOG" || rc=$?
kv dataset list --subset "$SUBSET" 2>&1 | tee -a "$LOG" || true
exit "$rc"
