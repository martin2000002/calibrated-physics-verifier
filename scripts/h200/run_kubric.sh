#!/usr/bin/env bash
# Builds the Kubric synthetic dataset on the H200 through udocker (no Docker, no root).
# Usage: bash scripts/h200/run_kubric.sh [subset]   (default: full; resumes when run again)
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ENV_FILE="${PHYSVERIFY_H200_ENV:-/tmp/mmonterot/env.sh}"
if [[ -f "$ENV_FILE" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set -u
fi

SUBSET="${1:-full}"
PROFILE="${PHYSVERIFY_KUBRIC_PROFILE:-precise}"
KUBRIC_THREADS="${PHYSVERIFY_KUBRIC_THREADS:-16}"
WORKERS="${PHYSVERIFY_KUBRIC_WORKERS:-12}"
JOBS="${PHYSVERIFY_KUBRIC_JOBS:-8}"
MIN_FREE_GB="${PHYSVERIFY_KUBRIC_MIN_FREE_GB:-50}"
UDOCKER="${PHYSVERIFY_UDOCKER:-/tmp/mmonterot/udocker-venv/bin/udocker}"
CONTAINER="${PHYSVERIFY_UDOCKER_CONTAINER:-kub}"
IMAGE="${PHYSVERIFY_KUBRIC_IMAGE:-kubricdockerhub/kubruntu:latest}"
DATA_DIR="${PHYSVERIFY__PATHS__DATA_DIR:-/tmp/mmonterot/kubric-data}"

for name in KUBRIC_THREADS WORKERS JOBS MIN_FREE_GB; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got '${!name}'"
done

# Datadog's ddtrace in the host Python breaks udocker unless it is switched off.
export DD_TRACE_ENABLED=false DD_SYMBOL_DATABASE_UPLOAD_ENABLED=false
export DD_INSTRUMENTATION_TELEMETRY_ENABLED=false DD_REMOTE_CONFIGURATION_ENABLED=false
export PHYSVERIFY__CONTAINER__ENGINE=udocker
export PHYSVERIFY__CONTAINER__UDOCKER_BIN="$UDOCKER"
export PHYSVERIFY__CONTAINER__UDOCKER_CONTAINER="$CONTAINER"
export PHYSVERIFY__PATHS__DATA_DIR="$DATA_DIR"
# A Kubric-only lock: a second build exits 75 at once and GPU jobs are not queued behind it.
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="${PHYSVERIFY_KUBRIC_LOCK:-/tmp/mmonterot/kubric.lock}"
export PHYSVERIFY__RUNTIME__LOCK_TIMEOUT_S=0

kv() {
    uv run --no-sync physverify --profile "$PROFILE" \
        --set "runtime.threads=$KUBRIC_THREADS" \
        --set "dataset.render.workers=$WORKERS" \
        --set "dataset.render.jobs_per_process=$JOBS" "$@"
}

[[ -x "$UDOCKER" ]] || die "udocker not found at $UDOCKER (set PHYSVERIFY_UDOCKER)"
if ! "$UDOCKER" inspect "$CONTAINER" >/dev/null 2>&1; then
    "$UDOCKER" pull "$IMAGE" || die "udocker pull $IMAGE failed"
    "$UDOCKER" create --name="$CONTAINER" "$IMAGE" || die "udocker create $CONTAINER failed"
fi
"$UDOCKER" run --nobanner "$CONTAINER" python3 -c "import kubric, bpy, pybullet" \
    || die "the $CONTAINER container cannot import kubric, bpy and pybullet"

mkdir -p "$DATA_DIR/runs"
FREE_GB="$(df --output=avail -B1G "$DATA_DIR" | tail -n 1 | tr -d ' ')" || die "df failed on $DATA_DIR"
[[ "$FREE_GB" =~ ^[0-9]+$ ]] || die "could not read the free space of $DATA_DIR"
((FREE_GB >= MIN_FREE_GB)) || die "only ${FREE_GB} GB free on $DATA_DIR, need $MIN_FREE_GB"

cd "$REPO_ROOT"
DOCTOR="$(kv doctor)" || die "physverify doctor failed"
grep -q '^container_engine: udocker' <<<"$DOCTOR" \
    || die "physverify doctor does not report container_engine: udocker"

LOG="$DATA_DIR/runs/kubric-$SUBSET-$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$(basename "$LOG")" "$DATA_DIR/runs/kubric-latest.log"
rc=0
kv dataset build --subset "$SUBSET" 2>&1 | tee "$LOG" || rc=$?
kv dataset list --subset "$SUBSET" 2>&1 | tee -a "$LOG" || true
exit "$rc"
