#!/usr/bin/env bash
# Behavior-equivalence harness: run the same CPU CLI scenarios on two revisions and diff the
# normalized stdout, stderr, exit codes and written files.
#
#   run_equivalence.sh <left-rev> <right-rev>     e.g. 9eb5888 HEAD
#
# Output under $EQ_WORKDIR (default $TMPDIR/physverify-equivalence): out/<label>/ (raw run dirs), norm/<label>/ (normalized), diff.txt, golden.txt.
# EQ_EXPECTED: space-separated file names under expected/ whose concatenated lines are the expected
# normalized diff (default: every expected/*.diff; set it empty to expect no difference).
# Exit 0 only when the sorted normalized diff equals the sorted expected lines and the golden blob
# lists are identical.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${EQ_REPO:-$(cd "$HERE/../.." && pwd)}"
WORK="${EQ_WORKDIR:-${TMPDIR:-/tmp}/physverify-equivalence}"
PY="$REPO/.venv/bin/python"
WT="$WORK/wt"
RUN="$WORK/run"
OUT="$WORK/out"
NORM="$WORK/norm"

[[ $# -eq 2 ]] || { sed -n '5,6p' "$0" >&2; exit 2; }
EXPECTED_DIR="$HERE/expected"
if [[ -v EQ_EXPECTED ]]; then
    read -r -a EXPECTED <<<"$EQ_EXPECTED"
else
    EXPECTED=()
    for file in "$EXPECTED_DIR"/*.diff; do
        [[ -e "$file" ]] && EXPECTED+=("$(basename "$file")")
    done
fi
LEFT="$1" RIGHT="$2"

worktree() { # <dir> <rev>
    git -C "$REPO" worktree remove --force "$1" 2>/dev/null || rm -rf "$1"
    git -C "$REPO" worktree add --detach --quiet "$1" "$2"
}

drop_worktree() {
    git -C "$REPO" worktree remove --force "$1" 2>/dev/null || rm -rf "$1"
    git -C "$REPO" worktree prune
}

# Test helper directories (calibration_cases, reporting_cases, check_scenarios, synthetic_clip)
# before and after the tests mirror src; a directory missing in one checkout is skipped by Python.
HELPER_DIRS=(fixtures calibration reporting checks selective_risk verification human_agreement
    generator_comparison scripts datasets)

py() { # <worktree> <args...>
    local tree="$1" path dir
    shift
    path="$tree/src"
    for dir in "${HELPER_DIRS[@]}"; do
        path+=":$tree/tests/$dir"
    done
    env -i HOME="$RUN/home" PATH="$PATH" LANG=C.UTF-8 COLUMNS=120 TERM=dumb NO_COLOR=1 \
        PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 EQ_WORKTREE="$tree" PYTHONPATH="$path" \
        "$PY" "$HERE/driver.py" "$@"
}

configs() { # <worktree>: the checkout's configs without the local overrides, plus a test lock
    local target="$RUN/repo/configs"
    mkdir -p "$target"
    cp -a "$1/configs/." "$target/"
    rm -f "$target"/local*.toml
    printf '[runtime]\nheavy_lock = "%s"\n' "$RUN/locks/heavy.lock" >"$target/local.toml"
}

run_side() { # <label> <rev>
    local label="$1" rev="$2" commit failed=0 name
    commit="$(git -C "$REPO" rev-parse "$rev")"
    rm -rf "$RUN" "$OUT/$label"
    mkdir -p "$RUN/home" "$RUN/locks"
    worktree "$WT/side" "$commit"
    configs "$WT/side"
    py "$WT/side" prepare "$RUN"
    mkdir -p "$RUN/_capture"
    while read -r name; do
        py "$WT/side" scenario "$name" "$RUN" </dev/null \
            >"$RUN/_capture/$name.out" 2>"$RUN/_capture/$name.err" || {
            echo "[$label] scenario $name failed:" >&2
            tail -n 5 "$RUN/_capture/$name.err" >&2
            failed=1
        }
    done < <(py "$WT/side" list)
    drop_worktree "$WT/side"
    mkdir -p "$OUT"
    mv "$RUN" "$OUT/$label"
    py "$REPO" normalize "$OUT/$label" "$NORM/$label" "$RUN" "$WT/side" \
        "$(git -C "$REPO" rev-parse "$LEFT")" "$(git -C "$REPO" rev-parse "$RIGHT")"
    return "$failed"
}

goldens() { # <rev>
    git -C "$REPO" ls-tree -r "$1" -- tests | awk '$4 ~ /golden\/|synth\/data\//{print $3}' | sort
}

mkdir -p "$WORK"
status=0
run_side left "$LEFT" || status=1
run_side right "$RIGHT" || status=1

expected_lines() {
    local name
    for name in "${EXPECTED[@]}"; do
        cat "$EXPECTED_DIR/$name"
    done
}

diff_status=0
(cd "$NORM" && diff -r left right) >"$WORK/diff.txt" || diff_status=$?
if ((diff_status > 1)); then
    echo "diff -r failed (exit $diff_status)" >&2
    status=1
elif [[ ! -s "$WORK/diff.txt" && ${#EXPECTED[@]} -eq 0 ]]; then
    echo "normalized outputs identical ($(find "$NORM/left" -type f | wc -l) files)"
elif diff <(sort "$WORK/diff.txt") <(expected_lines | sort) >"$WORK/unexpected.txt"; then
    echo "normalized outputs differ only as expected (${EXPECTED[*]}; $(wc -l <"$WORK/diff.txt") lines)"
else
    echo "normalized outputs DIFFER from expected (${EXPECTED[*]:-none}): see $WORK/diff.txt and $WORK/unexpected.txt ($(wc -l <"$WORK/unexpected.txt") lines)"
    status=1
fi

if diff <(goldens "$LEFT") <(goldens "$RIGHT") >"$WORK/golden.txt"; then
    echo "golden blobs identical ($(goldens "$LEFT" | wc -l) blobs)"
else
    echo "golden blobs DIFFER: see $WORK/golden.txt"
    status=1
fi
exit "$status"
