#!/usr/bin/env bash
# P1 (ADR 0014): depth (A rows) and detection (B rows) configurations on the Kubric synthetic set.
# Usage: run_p1_perception.sh v1|v1-precise [ID ...]   (default IDs: every row but A-392)
# Each configuration gets its own calib error profile, then extract, stage errors, per-clip
# collection, end-to-end and verify; extract, diagnose and verify exit 1 when some clips fail,
# which is reported, not fatal. Data live apart from every other job, under $P1_DATA.
# The pinned thresholds must score something: an empty "checks" table stops the batch unless
# PHYSVERIFY_P1_PROVISIONAL=1, which verifies with the provisional configs/checks.toml thresholds
# and records that in the manifest.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ROOT="${PHYSVERIFY_ROOT:-$REPO_ROOT}"
P1_DATA="${PHYSVERIFY_P1_DATA:-$H200_ROOT/p1/data}"
P1_HF="${PHYSVERIFY_P1_HF:-$H200_ROOT/p1/hf}"
SEED="$H200_ROOT/Repo/data"
SEED_HF="$H200_ROOT/.cache/huggingface"
PRECISE_DATA="$H200_ROOT/kubric-data/synthetic/v1-precise"
PROVISIONAL="${PHYSVERIFY_P1_PROVISIONAL:-0}"
DEFAULT_IDS=(A-fast A-518 A-280 A-e2 A-met B-tpq B-bj B-bpq)
MANIFEST="$P1_DATA/runs/p1-manifest.tsv"
LOGS="$P1_DATA/runs/p1-logs"

# The root --set options of one configuration, one per line; dies on an unknown ID.
label_sets() {
    case "$1" in
        A-fast | A-518) ;;
        A-280) echo "extraction.providers.depth.vda_small.input_size=280" ;;
        A-392) echo "extraction.providers.depth.vda_small.input_size=392" ;;
        A-e2) echo "extraction.providers.lift.backproject.erode_px=2" ;;
        A-met) echo "extraction.plan.depth=vda_metric_small" ;;
        B-tpq) echo "extraction.plan.detect=grounding_dino_tiny_per_query" ;;
        B-bj)
            echo "extraction.plan.detect=grounding_dino_base"
            echo "extraction.providers.detect.grounding_dino_base.prompt_mode=joint"
            ;;
        B-bpq) echo "extraction.plan.detect=grounding_dino_base" ;;
        *) die "unknown P1 configuration '$1' (A-fast A-518 A-280 A-392 A-e2 A-met B-tpq B-bj B-bpq)" ;;
    esac
}

# A-fast is the baseline of both tables; the B rows change detection only, so they stay on fast.
profile_of() {
    case "$1" in
        A-fast | B-*) echo fast ;;
        *) echo precise ;;
    esac
}

dataset_sets() {
    case "$SET" in
        v1) printf '%s\n' dataset.version=v1 dataset.resolution=128 dataset.background=plain ;;
        v1-precise) printf '%s\n' dataset.version=v1-precise dataset.resolution=256 dataset.background=hdri ;;
    esac
}

# The selection after the subcommand: the local subset on v1; on v1-precise the full subset, or
# the clips listed in PHYSVERIFY_P1_CLIPS (the scenes never probed).
selection() {
    SELECTION=(--dataset kubric_synthetic)
    if [[ "$SET" == v1 ]]; then
        SELECTION+=(--subset local)
    elif [[ -n "${PHYSVERIFY_P1_CLIPS:-}" ]]; then
        local clip
        SELECTION+=(--subset full)
        for clip in $PHYSVERIFY_P1_CLIPS; do SELECTION+=(--clips "$clip"); done
    else
        SELECTION+=(--subset full)
    fi
}

# GLOBAL: the root options of configuration $1, every --set before any subcommand.
global_options() {
    local label="$1" item
    GLOBAL=(--profile "$(profile_of "$label")" --device cuda
        --set "runtime.threads=$THREADS"
        --set "runtime.heavy_lock=$P1_DATA/p1.lock"
        --set "paths.data_dir=$P1_DATA"
        --set "paths.hf_home=$P1_HF"
        --set "paths.error_profile=$P1_DATA/calibration/stage_errors.p1-$label.json")
    while IFS= read -r item; do GLOBAL+=(--set "$item"); done < <(dataset_sets)
    while IFS= read -r item; do GLOBAL+=(--set "$item"); done < <(label_sets "$label")
    # A thresholds path that does not exist makes verify fall back to configs/checks.toml.
    [[ "$PROVISIONAL" != 1 ]] || GLOBAL+=(--set "paths.thresholds=$P1_DATA/calibration/provisional.absent")
}

# Read-only seeding from the main data directory; an existing copy is kept as it is, so the
# pinned thresholds never move between batches. v1-precise is linked, not copied: manifest video
# paths are relative to paths.data_dir.
seed() {
    local dir
    mkdir -p "$P1_DATA/calibration" "$LOGS" "$P1_HF"
    for dir in synthetic weights vendor; do
        [[ -e "$P1_DATA/$dir" || ! -d "$SEED/$dir" ]] || cp -a "$SEED/$dir" "$P1_DATA/$dir"
    done
    if [[ "$SET" == v1-precise && ! -e "$P1_DATA/synthetic/v1-precise" ]]; then
        [[ -d "$PRECISE_DATA" ]] || die "no v1-precise dataset at $PRECISE_DATA"
        mkdir -p "$P1_DATA/synthetic"
        ln -s "$PRECISE_DATA" "$P1_DATA/synthetic/v1-precise"
    fi
    local thresholds="$P1_DATA/calibration/thresholds.json"
    if [[ ! -f "$thresholds" ]]; then
        [[ -f "$SEED/calibration/thresholds.perceived.fast.json" ]] \
            || die "no thresholds.perceived.fast.json under $SEED/calibration"
        cp "$SEED/calibration/thresholds.perceived.fast.json" "$thresholds"
    fi
    if [[ ! -e "$P1_HF/hub" && -d "$SEED_HF/hub" ]]; then
        cp -al "$SEED_HF/hub" "$P1_HF/hub" 2>/dev/null \
            || { rm -rf "$P1_HF/hub" && cp -a "$SEED_HF/hub" "$P1_HF/hub"; }
    fi
}

# The sha of the pinned table, or provisional:<sha of configs/checks.toml>; dies on a pinned
# table without checks, since every verdict would come out UNSCORED.
thresholds_sha() {
    local pinned="$P1_DATA/calibration/thresholds.json"
    if [[ "$PROVISIONAL" == 1 ]]; then
        echo "provisional:$(sha256sum "$ROOT/configs/checks.toml" | cut -d' ' -f1)"
        return
    fi
    python3 -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("checks") else 1)' \
        "$pinned" || die "$pinned has no checks (every verdict UNSCORED); set PHYSVERIFY_P1_PROVISIONAL=1"
    sha256sum "$pinned" | cut -d' ' -f1
}

sources_sha() {
    (cd "$ROOT" && find src -name '*.py' | LC_ALL=C sort | while IFS= read -r f; do sha256sum "$f"; done) \
        | sha256sum | cut -d' ' -f1
}

record() {
    local label="$1" name="$2" log="$3" run_dir
    run_dir="$(sed -n '/^run: /{s///p;q}' "$log")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SET-$label" "$name" \
        "${run_dir:--}" "${CUDA_VISIBLE_DEVICES:-unset}" "$SRC_SHA" "$THRESHOLDS_SHA" >>"$MANIFEST"
}

# step LABEL NAME TOLERATED -- COMMAND...: runs COMMAND with its output in the label's log;
# exits with the command's code unless it is 0 or TOLERATED.
step() {
    local label="$1" name="$2" tolerated="$3"
    shift 4
    local log="$LOGS/$SET-$label-$name.log"
    set +e
    "$@" 2>&1 | tee "$log"
    local code="${PIPESTATUS[0]}"
    set -e
    record "$label" "$name" "$log"
    if [[ "$code" -ne 0 && "$code" -ne "$tolerated" ]]; then
        echo "step failed ($code): $SET-$label $name; output in $log" >&2
        exit "$code"
    fi
    [[ "$code" -eq 0 ]] || echo "some clips failed: $SET-$label $name" >&2
}

configuration() {
    local label="$1"
    global_options "$label"
    local pv=(uv run --no-sync physverify "${GLOBAL[@]}")
    step "$label" fetch 0 -- "${pv[@]}" extract fetch-weights
    step "$label" extract 1 -- "${pv[@]}" extract run "${SELECTION[@]}"
    step "$label" stages 1 -- "${pv[@]}" diagnose stages --write-profile "${SELECTION[@]}"
    step "$label" collect 0 -- "${pv[@]}" scene-stats per-clip --label "$label" \
        --out "$P1_DATA/runs/p1-$SET-$label.json" "${SELECTION[@]}"
    step "$label" end-to-end 1 -- "${pv[@]}" diagnose end-to-end "${SELECTION[@]}"
    step "$label" verify 1 -- "${pv[@]}" verify --level perceived "${SELECTION[@]}"
    if [[ "${PHYSVERIFY_P1_ATTRIBUTION:-0}" == 1 ]]; then
        step "$label" attribution 1 -- "${pv[@]}" diagnose attribution "${SELECTION[@]}"
    fi
}

SET="${1:-}"
[[ "$SET" == v1 || "$SET" == v1-precise ]] || die "usage: $0 v1|v1-precise [ID ...]"
shift
IDS=("$@")
[[ ${#IDS[@]} -gt 0 ]] || IDS=("${DEFAULT_IDS[@]}")
for id in "${IDS[@]}"; do label_sets "$id" >/dev/null; done

require_cuda
source_h200_env
cd "$ROOT"
seed
SRC_SHA="$(sources_sha)"
THRESHOLDS_SHA="$(thresholds_sha)"
selection
for id in "${IDS[@]}"; do configuration "$id"; done
echo "P1 $SET done: ${IDS[*]} (manifest: $MANIFEST)"
