#!/usr/bin/env bash
# Migrates a stored Kubric dataset to the position-consistent GT velocity (WP-G 1.C, ADR 0004)
# in place, CPU only, and audits the result. Resumes when run again.
#
#   1. Optionally copies a source dataset and cache in (read only on the source).
#   2. Keeps a copy of the dataset as it was (the audit's "before"), unless one exists.
#   3. `dataset migrate-velocity`: APPLY again on every clip; renders whose poses, sizes or hidden
#      frames changed are discarded and listed in <dataset>/migrations/velocity-position.json.
#   4. `dataset build --subset S` for every subset: renders the discarded clips and exports all.
#   5. `extract purge --clips-from <manifest>` on every cache root (perception caches of the
#      re-rendered clips; the cache key has no frame hash).
#   6. Oracle sweeps of every verify subset with fit_source=position (default) and =velocity.
#   7. `dataset audit-migration`: re-renders, frames/mp4, contacts, records, labels, rest cells,
#      stale cache, written to $OUT/audit.json.
#
# Required:
#   PHYSVERIFY_MIGRATE_ROOT     every path written (data, cache, lock, HOME) must be under it
#   PHYSVERIFY_MIGRATE_VERSION  dataset version to migrate (v1 for the default profile,
#                               v1-precise for precise)
# Optional:
#   PHYSVERIFY__PATHS__DATA_DIR      data dir holding synthetic/<version> (default: $ROOT/data)
#   PHYSVERIFY_MIGRATE_SOURCE_DATA   dataset directory copied into synthetic/<version> first
#   PHYSVERIFY_MIGRATE_SOURCE_CACHE  cache directory copied into paths.cache_dir first
#   PHYSVERIFY_MIGRATE_BEFORE        the copy taken before migrating (default: the source dataset
#                                    when given, else $DATA_DIR/migration-before/<version>)
#   PHYSVERIFY_MIGRATE_SUBSETS       subsets to build (default: full)
#   PHYSVERIFY_MIGRATE_VERIFY        subsets to sweep (default: the built ones)
#   PHYSVERIFY_MIGRATE_CACHE_ROOTS   extra cache roots to purge, space separated
#   PHYSVERIFY_KUBRIC_PROFILE        default (no --profile) or precise (default: precise)
#   PHYSVERIFY_KUBRIC_THREADS / _WORKERS / _JOBS   CPU budget of the render (16 / 4 / 4)
#   PHYSVERIFY_MIGRATE_PYTHON        interpreter that runs this checkout's src (default: uv run)
#
# Dev check of local and levels on a copy of v1:
#   PHYSVERIFY_MIGRATE_ROOT=/home/mmonterot/dev PHYSVERIFY_MIGRATE_VERSION=v1 \
#   PHYSVERIFY__PATHS__DATA_DIR=/home/mmonterot/dev/scratch/wpg1-data \
#   PHYSVERIFY_MIGRATE_SOURCE_DATA=/home/mmonterot/work/Repo/data/synthetic/v1 \
#   PHYSVERIFY_MIGRATE_SOURCE_CACHE=/home/mmonterot/work/Repo/data/cache \
#   PHYSVERIFY_MIGRATE_SUBSETS="local levels" PHYSVERIFY_KUBRIC_PROFILE=default \
#   PHYSVERIFY_MIGRATE_PYTHON=/home/mmonterot/dev/venv/bin/python PHYSVERIFY_MIGRATE_COMMIT=<sha> \
#   bash scripts/h200/migrate_velocity.sh
# Full migration of v1-precise in place:
#   PHYSVERIFY_MIGRATE_ROOT=/home/mmonterot/work PHYSVERIFY_MIGRATE_VERSION=v1-precise \
#   PHYSVERIFY__PATHS__DATA_DIR=/home/mmonterot/work/kubric-data \
#   PHYSVERIFY_MIGRATE_CACHE_ROOTS="/home/mmonterot/work/p1/data/cache" \
#   PHYSVERIFY_MIGRATE_SUBSETS="smoke local levels full" PHYSVERIFY_MIGRATE_VERIFY="local levels" \
#   bash scripts/h200/migrate_velocity.sh
set -euo pipefail

die() {
    printf '%s\n' "$*" >&2
    exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${PHYSVERIFY_MIGRATE_ROOT:?set PHYSVERIFY_MIGRATE_ROOT}"
VERSION="${PHYSVERIFY_MIGRATE_VERSION:?set PHYSVERIFY_MIGRATE_VERSION}"
ROOT="$(realpath -m "$ROOT")"
PROFILE="${PHYSVERIFY_KUBRIC_PROFILE:-precise}"
THREADS="${PHYSVERIFY_KUBRIC_THREADS:-16}"
WORKERS="${PHYSVERIFY_KUBRIC_WORKERS:-4}"
JOBS="${PHYSVERIFY_KUBRIC_JOBS:-4}"
SUBSETS="${PHYSVERIFY_MIGRATE_SUBSETS:-full}"
VERIFY="${PHYSVERIFY_MIGRATE_VERIFY:-$SUBSETS}"
SOURCE_DATA="${PHYSVERIFY_MIGRATE_SOURCE_DATA:-}"
SOURCE_CACHE="${PHYSVERIFY_MIGRATE_SOURCE_CACHE:-}"
for name in THREADS WORKERS JOBS; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got '${!name}'"
done

# Nothing is sourced from an env.sh: every location is set here and checked against $ROOT.
export HOME="$ROOT" XDG_CACHE_HOME="$ROOT/.cache" PHYSVERIFY_H200_ROOT="$ROOT"
export PHYSVERIFY__PATHS__DATA_DIR="${PHYSVERIFY__PATHS__DATA_DIR:-$ROOT/data}"
export PHYSVERIFY__RUNTIME__HEAVY_LOCK="${PHYSVERIFY_MIGRATE_LOCK:-$ROOT/migrate-velocity.lock}"
export PHYSVERIFY__RUNTIME__LOCK_TIMEOUT_S=0 PHYSVERIFY__CONTAINER__ENGINE=docker
export CUDA_VISIBLE_DEVICES=""

pv() {
    local profile=()
    [[ "$PROFILE" == default ]] || profile=(--profile "$PROFILE")
    local cli=(uv run --no-sync physverify)
    [[ -z "${PHYSVERIFY_MIGRATE_PYTHON:-}" ]] || cli=(env "PYTHONPATH=$REPO_ROOT/src"
        "$PHYSVERIFY_MIGRATE_PYTHON" -c "from physverify.cli.main import main; main()")
    "${cli[@]}" "${profile[@]}" --device cpu \
        --set "runtime.threads=$THREADS" \
        --set "dataset.render.workers=$WORKERS" \
        --set "dataset.render.jobs_per_process=$JOBS" "$@"
}

under_root() {
    local path
    path="$(realpath -m "$1")"
    [[ "$path" == "$ROOT" || "$path" == "$ROOT"/* ]]
}

doctor_value() {
    sed -n "s/^$1: //p" <<<"$DOCTOR" | sed 's/ (.*)$//'
}

cd "$REPO_ROOT"
DOCTOR="$(pv doctor)" || die "physverify doctor failed"
DATA_DIR="$(doctor_value paths.data_dir)"
CACHE_DIR="$(doctor_value paths.cache_dir)"
RUNS_DIR="$(doctor_value paths.runs_dir)"
SYNTH_DIR="$(doctor_value paths.synthetic_dir)"
LOCK="$(doctor_value heavy_lock)"
for path in "$DATA_DIR" "$CACHE_DIR" "$RUNS_DIR" "$SYNTH_DIR" "$LOCK"; do
    [[ -n "$path" ]] && under_root "$path" || die "doctor reports '$path' outside $ROOT"
done
CACHE_ROOTS=("$CACHE_DIR")
for path in ${PHYSVERIFY_MIGRATE_CACHE_ROOTS:-}; do
    under_root "$path" || die "cache root $path is outside $ROOT"
    CACHE_ROOTS+=("$path")
done
docker info >/dev/null 2>&1 || die "docker does not respond"

DATASET="$SYNTH_DIR/$VERSION"
MANIFEST="$DATASET/migrations/velocity-position.json"
OUT="$DATA_DIR/runs/migrate-velocity-$VERSION"
mkdir -p "$OUT"
LOG="$OUT/migrate-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
printf '%s\n' "$DOCTOR"

if [[ -n "$SOURCE_DATA" && ! -f "$MANIFEST" ]]; then
    mkdir -p "$DATASET"
    rsync -a --delete "$SOURCE_DATA/" "$DATASET/"
fi
if [[ -n "$SOURCE_CACHE" && ! -f "$MANIFEST" ]]; then
    mkdir -p "$CACHE_DIR"
    rsync -a "$SOURCE_CACHE/" "$CACHE_DIR/"
fi
[[ -f "$DATASET/plan.json" ]] || die "no dataset at $DATASET"

BEFORE="${PHYSVERIFY_MIGRATE_BEFORE:-${SOURCE_DATA:-$DATA_DIR/migration-before/$VERSION}}"
if [[ ! -d "$BEFORE" ]]; then
    [[ ! -f "$MANIFEST" ]] || die "$MANIFEST exists but the copy taken before ($BEFORE) does not"
    under_root "$BEFORE" || die "$BEFORE is outside $ROOT"
    mkdir -p "$BEFORE"
    rsync -a "$DATASET/" "$BEFORE/"
fi

COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "${PHYSVERIFY_MIGRATE_COMMIT:?no git checkout: set PHYSVERIFY_MIGRATE_COMMIT}")"
pv dataset migrate-velocity --commit "$COMMIT"
for subset in $SUBSETS; do
    pv dataset build --subset "$subset"
    pv dataset list --subset "$subset"
done
purge=()
for root in "${CACHE_ROOTS[@]}"; do
    purge+=(--cache-root "$root")
done
pv extract purge --clips-from "$MANIFEST" "${purge[@]}"

tables=()
for subset in $VERIFY; do
    for source in position velocity; do
        sweep="$OUT/sweep-$subset-$source.txt"
        pv --set "checks.R3a.fit_source=$source" --set "checks.R3b.fit_source=$source" \
            verify --dataset kubric_synthetic --subset "$subset" --level oracle --sweep \
            >"$sweep" 2>&1 || true
        cat "$sweep"
        run_dir="$(sed -n 's/^run: //p' "$sweep")"
        [[ -n "$run_dir" ]] || die "no run directory in $sweep"
        [[ "$source" == position ]] && tables+=(--table "$run_dir/table.csv")
    done
done
pv dataset audit-migration --before "$BEFORE" --out "$OUT/audit.json" "${purge[@]}" "${tables[@]}"
