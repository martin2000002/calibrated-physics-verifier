#!/usr/bin/env bash
# H2 (ADR 0011): dev pass (fetch, extract, verify, freeze) then test pass (fetch, extract, verify,
# compare) on VideoPhy. Needs the recorded caption review in annotations/videophy_subset.json.
# Extract and verify exit 1 when some clips fail; those clips are counted by compare, not fatal.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"

H2_LINK="h2-latest"
DEV_SUBSET="${PHYSVERIFY_H2_DEV:-h2_dev}"
TEST_SUBSET="${PHYSVERIFY_H2_TEST:-h2_test}"

# step TOLERATED -- ARGS: runs physverify ARGS with its output also in $STEP_LOG; exits with the
# command's code unless it is 0 or TOLERATED.
step() {
    local tolerated="$1"
    shift 2
    STEP_LOG="$(mktemp "${TMPDIR:-/tmp}/h2-step.XXXXXX.log")"
    set +e
    pv "$@" 2>&1 | tee "$STEP_LOG"
    local code="${PIPESTATUS[0]}"
    set -e
    if [[ "$code" -ne 0 && "$code" -ne "$tolerated" ]]; then
        echo "step failed ($code): physverify $*; output in $STEP_LOG" >&2
        exit "$code"
    fi
    [[ "$code" -eq 0 ]] || echo "some clips failed: physverify $*" >&2
}

# The run directory the last step printed; the step's log is moved next to it.
run_of() {
    local run_dir
    run_dir="$(sed -n '/^run: /{s///p;q}' "$STEP_LOG")"
    [[ -n "$run_dir" && -d "$run_dir" ]] || die "no run directory in $STEP_LOG"
    mv "$STEP_LOG" "$run_dir.log"
    printf '%s\n' "$run_dir"
}

pass() {
    local subset="$1"
    step 0 -- agreement fetch --subset "$subset"
    step 1 -- extract run --dataset videophy --subset "$subset"
    run_of >/dev/null
    step 1 -- verify --dataset videophy --level perceived --subset "$subset"
    VERIFY_RUN="$(run_of)"
}

require_cuda
bash scripts/h200/setup.sh
pv agreement subset --check || die "record the caption review first (agreement subset --record)"

pass "$DEV_SUBSET"
# On a re-run, freeze keeps an unchanged pre-registration and exits 2 if the frozen state moved.
step 0 -- agreement freeze --verify-run "$VERIFY_RUN"
PREREG="$(sed -n '/^preregistration: /{s///p;q}' "$STEP_LOG")"
[[ -n "$PREREG" && -f "$PREREG" ]] || die "no pre-registration in $STEP_LOG"
mv "$STEP_LOG" "$VERIFY_RUN/freeze.log"

pass "$TEST_SUBSET"
step 0 -- agreement compare --verify-run "$VERIFY_RUN" --subset "$TEST_SUBSET"
mv "$STEP_LOG" "$VERIFY_RUN/agreement.log"
mkdir -p "$VERIFY_RUN/agreement"
cp "$PREREG" "$VERIFY_RUN/agreement/preregistration.json"
ln -sfn "$(basename "$VERIFY_RUN")" "$(dirname "$VERIFY_RUN")/$H2_LINK"
echo "H2 run: $(dirname "$VERIFY_RUN")/$H2_LINK (summary: agreement/summary.md)"
