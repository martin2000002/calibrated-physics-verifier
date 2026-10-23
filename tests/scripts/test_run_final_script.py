"""scripts/h200/run_final.sh against a fake interpreter, ``git`` and Kubric render."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from shell_sandbox import Sandbox, executable

REPO = Path(__file__).resolve().parents[2]
H200 = REPO / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "_steps.sh", "_guards.sh", "run_final.sh", "freeze_query.py")
TOOLS = (
    "awk", "basename", "bash", "cat", "cmp", "cp", "cut", "date", "diff", "dirname", "env",
    "find", "flock", "grep", "head", "mkdir", "mv", "nice", "nproc", "realpath", "rm", "rmdir",
    "sed", "sha256sum", "sort", "tar", "tee", "touch", "xargs",
)  # fmt: skip
HEAD = "c0ffee1234567" + "0" * 27
V1 = "a" * 40
PREFIX = re.compile(r"^--profile precise --device cpu --set runtime\.threads=\d+ ")
# freeze_query.py field and seed run for real (they only read freeze.json or hash strings); the
# ledger is a file of commit|sets|run_dir lines that a fake look appends to before it computes,
# as the sealed guard does, and that the fake look-state and v1-state read with the one-look rule.
FAKE_PYTHON = r"""#!/bin/bash
LEDGER_FILE="${PHYSVERIFY__SEALED__LEDGER:-$LEDGER}"
log_env() {
    printf 'env %s|%s|%s|%s|%s|%s|%s\n' "${CUDA_VISIBLE_DEVICES-unset}" \
        "${PHYSVERIFY_UNSEAL-unset}" "${PHYSVERIFY_UNSEAL_REPEAT-unset}" \
        "$PHYSVERIFY__PATHS__CACHE_DIR" "$PHYSVERIFY__PATHS__CALIBRATION_DIR" \
        "$PHYSVERIFY__PATHS__RUNS_DIR" "$PHYSVERIFY__RUNTIME__HEAVY_LOCK" >> "$CALLS"
}
new_run() {
    local n
    n=$(( $(cat "$PHYSVERIFY__PATHS__RUNS_DIR/counter" 2>/dev/null || echo 0) + 1 ))
    mkdir -p "$PHYSVERIFY__PATHS__RUNS_DIR/r$n"
    echo "$n" > "$PHYSVERIFY__PATHS__RUNS_DIR/counter"
    RUN="$PHYSVERIFY__PATHS__RUNS_DIR/r$n"
    echo "run: $RUN"
}
covering() {
    local line sets s
    [[ -f "$LEDGER_FILE" ]] || return 0
    while IFS= read -r line; do
        sets="$(cut -d '|' -f 2 <<<"$line")"
        for s in ${1//,/ }; do
            if [[ ",$sets," == *",$s,"* ]]; then echo "$line"; break; fi
        done
    done < "$LEDGER_FILE"
}
evaluate() {
    new_run
    if [[ -n "${PHYSVERIFY_UNSEAL:-}" ]]; then
        [[ -f "$LEDGER_FILE" ]] || { echo "no sealed ledger" >&2; exit 2; }
        echo "$1|$PHYSVERIFY_UNSEAL|$RUN" >> "$LEDGER_FILE"
    fi
    if [[ -n "${EVAL_FAIL_ONCE:-}" && ! -e "$EVAL_FAIL_ONCE" ]]; then
        touch "$EVAL_FAIL_ONCE"; exit 3
    fi
    echo '{}' > "$RUN/evaluation.json"
    [[ -z "${SKIPPED:-}" ]] || echo "real dataset objectron_static skipped: no extracted clip"
    echo "evaluation: $RUN/evaluation.json"
}
if [[ "$1" != -c ]]; then
    script="$(basename "$1")"
    printf 'query %s %s\n' "$script" "${*:2}" >> "$CALLS"
    case "$script $2" in
      "freeze_query.py field" | "freeze_query.py seed") exec "$REAL_PYTHON" "$@" ;;
      "freeze_query.py artifacts") exit "${ARTIFACTS_EXIT:-0}" ;;
      "freeze_query.py render-env")
        echo "PHYSVERIFY__DATASET__VERSION=v-$3"; echo "PHYSVERIFY__DATASET__SEED=$4" ;;
      "freeze_query.py plan")
        version="v-$3"; [[ "$3" != kubric_fresh_blind ]] || version=v-kubric_fresh
        [[ -f "$PLANS/$version" ]] || { echo "freeze_query: no plan: render first" >&2; exit 2; }
        clips="${PLAN_CLIPS:-4}"
        echo "plan $PLANS/$version: version $version, seed $4, 10 scenes, $clips clips, 0 calib" ;;
      "freeze_query.py look-state")
        shift 2; retry=""
        if [[ "$1" == --retry ]]; then retry=1; shift; fi
        specs="$*"
        [[ -f "$LEDGER_FILE" ]] || { echo "freeze_query: no sealed ledger" >&2; exit 2; }
        mapfile -t found < <(covering "${specs// /,}")
        if [[ -z "$retry" ]]; then
            [[ ${#found[@]} -eq 0 ]] || { echo "freeze_query: already looked at" >&2; exit 2; }
            echo first
        else
            [[ ${#found[@]} -eq 1 ]] || { echo "freeze_query: --retry needs one" >&2; exit 2; }
            [[ ! -f "${found[0]##*|}/evaluation.json" ]] \
                || { echo "freeze_query: the earlier look wrote evaluation.json" >&2; exit 2; }
            echo retry
        fi ;;
      "freeze_query.py v1-state")
        shift 3; specs="$*"
        found="$(covering "${specs// /,}")"
        grep -q "^$HEAD_SHA|" <<<"$found" || { echo "freeze_query: no look yet" >&2; exit 2; }
        ! grep -q "^$V1_SHA|" <<<"$found" || { echo "freeze_query: v1 arm done" >&2; exit 2; }
        echo v1arm ;;
      "freeze_query.py ledger")
        [[ -f "$LEDGER_FILE" ]] || exit 2
        echo "ledger $LEDGER_FILE: $(grep -c . "$LEDGER_FILE") lines" ;;
      "run_query.py clips") printf '%s-1\n%s-2\n%s-3\n' "$3" "$3" "$3" ;;
      "overlay_cli.py "*)
        log_env
        [[ -z "${META:-}" ]] || { mkdir -p "$2/x"; touch "$2/x/meta.json"; }
        evaluate "$V1_SHA" ;;
    esac
    exit 0
fi
shift 2
printf 'pv %s\n' "$*" >> "$CALLS"
log_env
case "$*" in
  *" doctor")
    CAL="$PHYSVERIFY__PATHS__CALIBRATION_DIR"
    echo "profile: $2"
    echo "device: $4"
    echo "config_dir: $REPO_CONFIGS"
    echo "paths.cache_dir: $PHYSVERIFY__PATHS__CACHE_DIR (env)"
    echo "paths.raw_dir: $PHYSVERIFY__PATHS__DATA_DIR/raw"
    echo "paths.runs_dir: $PHYSVERIFY__PATHS__RUNS_DIR"
    echo "paths.calibration_dir: $CAL"
    echo "paths.error_profile: $CAL/stage_errors.precise.json"
    echo "paths.thresholds: $CAL/thresholds.json" ;;
  *" extract run "*)
    dataset="$(sed -n 's/.*--dataset \([^ ]*\).*/\1/p' <<<"$*")"
    mkdir -p "$PHYSVERIFY__PATHS__CACHE_DIR/$dataset" ;;
  *" download fetch "*)
    name="$(sed -n 's/.* download fetch \([^ ]*\).*/\1/p' <<<"$*")"
    echo "fetch $name seed ${PHYSVERIFY__OBJECTRON_STATIC__SEED-unset}" >> "$CALLS"
    mkdir -p "$PHYSVERIFY__PATHS__DATA_DIR/raw/$name/sub"
    echo a > "$PHYSVERIFY__PATHS__DATA_DIR/raw/$name/a.mp4"
    echo b > "$PHYSVERIFY__PATHS__DATA_DIR/raw/$name/sub/b.json" ;;
  *" calibrate evaluate "*) evaluate "$HEAD_SHA" ;;
esac
exit 0
"""
FAKE_GIT = """#!/bin/bash
case "$*" in
  "-C $CKV rev-parse HEAD") echo "${CKV_SHA:-$V1_SHA}" ;;
  "-C $CKV status --porcelain "*) printf '%s' "${CKV_DIRTY:-}" ;;
  *" rev-parse HEAD") echo "$HEAD_SHA" ;;
  *" status --porcelain "*) printf '%s' "${DIRTY_FILES:-}" ;;
  *" archive HEAD configs") tar -c -C "$2" configs ;;
esac
"""
FAKE_KUBRIC = """#!/bin/bash
printf 'kubric %s|%s|%s|%s\\n' "$*" "$PHYSVERIFY__DATASET__VERSION" "$PHYSVERIFY__DATASET__SEED" \\
    "$UV_PROJECT_ENVIRONMENT" >> "$CALLS"
mkdir -p "$PLANS"
touch "$PLANS/$PHYSVERIFY__DATASET__VERSION"
"""
FORBIDDEN = '#!/bin/bash\necho "forbidden $(basename "$0") $*" >> "$CALLS"\n'
BASE_FILES = (
    "calibration.json",
    "calibration/stage_errors.precise.json",
    "calibration/stage_sigma.json",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(calibration: Path, exclusions: Path) -> int:
    """Amendment 2 section 2, written out independently of freeze_query.py."""
    joined = HEAD + _sha(calibration) + _sha(exclusions)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8], 16)


REPO_CONFIG_FILES = {
    "default.toml": "default\n",
    "verifier/checks.toml": "frozen checks\n",
    "studies/fusion.toml": "fusion\n",
}


class FinalSandbox(Sandbox):
    def __init__(self, root: Path) -> None:
        super().__init__(root.resolve(), SCRIPT_NAMES, TOOLS)
        self.ledger = self.root / "ledger.jsonl"
        self.ledger.touch()
        for name, text in REPO_CONFIG_FILES.items():
            (self.repo / "configs" / name).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / "configs" / name).write_text(text, encoding="utf-8")
        executable(self.scripts / "run_kubric.sh", FAKE_KUBRIC)
        executable(self.fakes / "python", FAKE_PYTHON)
        executable(self.fakes / "git", FAKE_GIT)
        for name in ("uv", "nvidia-smi"):
            executable(self.fakes / name, FORBIDDEN)
        self.ckv = self.root / "ckv"
        (self.ckv / "configs").mkdir(parents=True)
        (self.ckv / "configs" / "checks.toml").write_text("v1 checks\n", encoding="utf-8")
        self.base = self.root / "base"
        for name in BASE_FILES:
            (self.base / name).parent.mkdir(parents=True, exist_ok=True)
            (self.base / name).write_text(f"base {name}", encoding="utf-8")
        overlay = self.root / "overlay_cli.py"
        overlay.write_text("", encoding="utf-8")
        self.final = self.root / "final"
        self.seed = self.freeze(self.final)
        self.env.update(
            {
                "REAL_PYTHON": sys.executable,
                "REPO_CONFIGS": str(self.repo / "configs"),
                "PHYSVERIFY_H200_ROOT": str(self.root),
                "PHYSVERIFY_H200_ENV": str(self.root / "no-env.sh"),
                "HEAD_SHA": HEAD,
                "V1_SHA": V1,
                "CKV": str(self.ckv),
                "LEDGER": str(self.ledger),
                "PLANS": str(self.root / "plans"),
                "PHYSVERIFY_FINAL_EXPECT_COMMIT": "c0ffee1",
                "PHYSVERIFY_FINAL_SEED": str(self.seed),
                "PHYSVERIFY_FINAL_PYTHON": str(self.fakes / "python"),
                "PHYSVERIFY_FINAL_V1_CHECKOUT": str(self.ckv),
                "PHYSVERIFY_FINAL_V1_BASE": str(self.base),
                "PHYSVERIFY_FINAL_OVERLAY_CLI": str(overlay),
            }
        )

    def freeze(self, final: Path, **change: Any) -> int:
        """A frozen directory as run_freeze.sh leaves it; returns the seed of the formula."""
        calibration = final / "calibration"
        calibration.mkdir(parents=True, exist_ok=True)
        for name in ("calibration.json", "exclusions.json", "stage_errors.precise.json"):
            (calibration / name).write_text(f"frozen {name}", encoding="utf-8")
        seed = _seed(calibration / "calibration.json", calibration / "exclusions.json")
        data: dict[str, Any] = {
            "commit": HEAD,
            "device": "cpu",
            "profile": "precise",
            "calibration_dir": str(calibration),
            "seed": seed,
            "artifacts": {
                name: {"path": str(calibration / f"{name}.json")}
                for name in ("calibration", "exclusions")
            },
            "v1": {"commit": V1, "calibration_sha256": _sha(self.base / "calibration.json")},
            **change,
        }
        (final / "freeze.json").write_text(json.dumps(data), encoding="utf-8")
        return seed

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call("run_final.sh", *args, **env)

    def ok(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        result = self.run(*args, **env)
        assert result.returncode == 0, result.stderr
        return result

    def commands(self) -> list[str]:
        return [PREFIX.sub("", line[3:]) for line in self.lines() if line.startswith("pv ")]

    def looks(self) -> list[str]:
        return [c for c in self.commands() if c.startswith("calibrate evaluate")]

    def queries(self) -> list[str]:
        return [line[6:] for line in self.lines() if line.startswith("query ")]

    def envs(self) -> list[list[str]]:
        return [line[4:].split("|") for line in self.lines() if line.startswith("env ")]

    def ledger_lines(self, path: Path | None = None) -> list[str]:
        return (path or self.ledger).read_text(encoding="utf-8").splitlines()

    def summary(self, name: str, final: Path | None = None) -> str:
        return ((final or self.final) / "run-final" / f"summary-{name}.md").read_text("utf-8")


@pytest.fixture
def sandbox(tmp_path: Path) -> FinalSandbox:
    return FinalSandbox(tmp_path)


def _look(final: Path, *dataset: str) -> str:
    return (
        f"calibrate evaluate --artifact {final / 'calibration' / 'calibration.json'} "
        + " ".join(dataset)
    )


FRESH = ("--dataset", "kubric_fresh", "--subset", "eval", "--no-real")
REAL_NEW = (
    "--dataset", "kubric_synthetic", "--subset", "calib", "--real-dataset", "physics_iq_views",
    "--real-dataset", "objectron_static",
)  # fmt: skip


def test_the_script_parses_and_is_executable() -> None:
    subprocess.run(["bash", "-n", str(H200 / "run_final.sh")], check=True)
    assert os.access(H200 / "run_final.sh", os.X_OK)


def test_the_fresh_kubric_look_and_its_v1_arm(sandbox: FinalSandbox) -> None:
    """The exported share noted, one ledger line per look plus the arm's, no meta.json, the
    arm's configs."""
    for args in (
        ("seed",),
        ("render", "kubric_fresh"),
        ("extract", "kubric_fresh"),
        ("look", "kubric_fresh"),
        ("v1arm", "kubric_fresh"),
    ):
        sandbox.ok(*args)
    final, arm = sandbox.final, sandbox.final / "v1arm"
    assert sandbox.commands() == [
        "doctor",
        "doctor",
        "doctor",
        "extract run --dataset kubric_fresh --subset eval",
        "doctor",
        _look(final, *FRESH),
        "doctor",
    ]
    (kubric,) = [line for line in sandbox.lines() if line.startswith("kubric ")]
    assert kubric == f"kubric full|v-kubric_fresh|{sandbox.seed}|{sandbox.root}"
    (overlay,) = [q for q in sandbox.queries() if q.startswith("overlay_cli.py")]
    assert overlay == (
        f"overlay_cli.py {arm / 'overlay'} --config-dir {arm / 'configs'} --profile precise "
        f"--device cpu --set runtime.threads=32 calibrate evaluate --artifact "
        f"{arm / 'calibration' / 'calibration.json'} " + " ".join(FRESH)
    )
    cache = str(sandbox.root / "kubric-data" / "cache")
    mine = [str(final / "calibration"), str(final / "runs"), str(final / "heavy.lock")]
    theirs = [str(arm / "calibration"), str(arm / "runs"), str(arm / "heavy.lock")]
    envs = sandbox.envs()
    assert envs[-1] == ["", "kubric_fresh", "1", cache, *theirs]
    assert envs[-3] == ["", "kubric_fresh", "unset", cache, *mine]
    assert all(e == ["", "unset", "unset", cache, *mine] for e in envs[:-3] + [envs[-2]])
    commits = [line.split("|")[:2] for line in sandbox.ledger_lines()]
    assert commits == [[HEAD, "kubric_fresh"], [V1, "kubric_fresh"]]
    assert (arm / "configs" / "checks.toml").read_text("utf-8") == "v1 checks\n"
    assert (arm / "configs" / "fusion.toml").read_text("utf-8") == "fusion\n"
    assert "checks.toml" in (arm / "configs.diff").read_text("utf-8")
    for name in BASE_FILES:
        adopted = arm / "calibration" / Path(name).name
        assert adopted.read_bytes() == (sandbox.base / name).read_bytes()
    assert "3 of 4 planned clips exported (75.0 %, threshold 90 %)" in sandbox.summary(
        "render-kubric_fresh"
    )
    assert "first look at kubric_fresh" in sandbox.summary("look-kubric_fresh")
    assert "overlay meta.json 0" in sandbox.summary("v1arm-kubric_fresh")
    assert not any("forbidden" in line for line in sandbox.lines())


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (("look", "kubric_fresh"), {"PHYSVERIFY_FINAL_SEED": "123"}, "is not the freeze's seed"),
        (("look", "kubric_fresh"), {"PHYSVERIFY_FINAL_SEED": ""}, "PHYSVERIFY_FINAL_SEED"),
        (("look", "kubric_fresh"), {"PHYSVERIFY_UNSEAL": "kubric_fresh"}, "PHYSVERIFY_UNSEAL is"),
        (("look", "kubric_fresh"), {"PHYSVERIFY_UNSEAL_REPEAT": "1"}, "PHYSVERIFY_UNSEAL_REPEAT"),
        (("look", "kubric_fresh"), {"HEAD_SHA": "d" * 40}, "expected c0ffee1"),
        (("look", "kubric_fresh"), {"DIRTY_FILES": " M src/x.py\n"}, "uncommitted or untracked"),
        (("look", "kubric_fresh"), {"PHYSVERIFY__SEALED__SETS": "[]"}, "PHYSVERIFY__SEALED__SETS"),
        (("look", "kubric_fresh"), {"PHYSVERIFY__SEALED__LEDGER": "/x"}, "does not own"),
        (("look", "kubric_fresh"), {"PHYSVERIFY_CONFIG_DIR": "/other"}, "PHYSVERIFY_CONFIG_DIR"),
        (("look", "kubric_fresh"), {"PHYSVERIFY_FINAL_FREEZE": "/no.json"}, "no freeze manifest"),
        (("look", "kubric_fresh"), {"ARTIFACTS_EXIT": "2"}, "the frozen files changed"),
        (("look", "kubric_fresh"), {"PHYSVERIFY_FINAL_DRYRUN": "2"}, "must be 0 or 1"),
        (("look", "kubric_fresh"), {"PHYSVERIFY_FINAL_DRYRUN": "1"}, "needs its own ledger"),
        (("look", "kubric_fresh"), {}, "the seed command has not run"),
        (("look",), {}, "usage"),
        (("look", "kubric_fresh", "--again"), {}, "usage"),
        (("look", "physics_iq_views"), {}, "look real_new"),
        (("render", "kubric_fresh_blind"), {}, "only kubric_fresh and kubric_fresh_ood"),
        (("fetch", "kubric_fresh"), {}, "only physics_iq_views and objectron_static"),
        (("extract", "real_new"), {}, "extracted as physics_iq_views"),
        (("v1arm", "kubric_fresh_ood"), {}, "the v1 arm runs on"),
        (("look", "kubric_synthetic"), {}, "unknown set"),
    ],
)
def test_a_refused_command_runs_nothing_and_leaves_the_ledger(
    sandbox: FinalSandbox, args: tuple[str, ...], env: dict[str, str], message: str
) -> None:
    """Every refusal exits 2 before any physverify call; the ledger keeps its lines."""
    sandbox.ledger.write_text("x|kubric_synthetic:calib|r0\n", encoding="utf-8")
    result = sandbox.run(*args, **env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == []
    assert sandbox.ledger_lines() == ["x|kubric_synthetic:calib|r0"]


def test_a_local_config_is_refused(sandbox: FinalSandbox) -> None:
    (sandbox.root / "repo" / "configs" / "local.toml").write_text("", encoding="utf-8")
    result = sandbox.run("seed")
    assert result.returncode == 2 and "local.toml exists" in result.stderr
    assert sandbox.commands() == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"commit": "b" * 40}, "freeze.json is of commit"),
        ({"device": "cuda"}, "not cpu, precise"),
        ({"profile": "fast"}, "not cpu, precise"),
        ({"calibration_dir": "/elsewhere"}, "froze /elsewhere"),
    ],
)
def test_a_freeze_of_another_state_is_refused(
    sandbox: FinalSandbox, change: dict[str, str], message: str
) -> None:
    sandbox.freeze(sandbox.final, **change)
    result = sandbox.run("seed")
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == []


def test_a_seed_off_the_formula_is_refused(sandbox: FinalSandbox) -> None:
    sandbox.freeze(sandbox.final, seed=12345)
    result = sandbox.run("seed", PHYSVERIFY_FINAL_SEED="12345")
    assert result.returncode == 2 and "formula gives" in result.stderr, result.stderr
    assert sandbox.commands() == ["doctor"]
    assert sandbox.run("render", "kubric_fresh", PHYSVERIFY_FINAL_SEED="12345").returncode == 2


def test_a_resumed_command_keeps_the_commit_and_the_mode(sandbox: FinalSandbox) -> None:
    sandbox.ok("seed")
    moved = sandbox.run("render", "kubric_fresh", HEAD_SHA="c0ffee1" + "9" * 33)
    assert moved.returncode == 2 and "freeze.json is of commit" in moved.stderr
    (sandbox.final / "freeze.json").write_text(
        (sandbox.final / "freeze.json").read_text("utf-8").replace(HEAD, "c0ffee1" + "9" * 33),
        encoding="utf-8",
    )
    moved = sandbox.run("render", "kubric_fresh", HEAD_SHA="c0ffee1" + "9" * 33)
    assert moved.returncode == 2 and "belongs to FINAL_COMMIT" in moved.stderr


def test_extract_needs_the_render_of_the_seed(sandbox: FinalSandbox) -> None:
    """No plan.json of this seed, no extraction."""
    sandbox.ok("seed")
    result = sandbox.run("extract", "kubric_fresh")
    assert result.returncode == 2 and "not the one of seed" in result.stderr
    assert sandbox.commands() == ["doctor", "doctor"]


def test_a_cache_written_before_the_first_extraction_is_refused(sandbox: FinalSandbox) -> None:
    """Another render or a dry run under the same name never feeds a look."""
    sandbox.ok("seed")
    sandbox.ok("render", "kubric_fresh")
    (sandbox.root / "kubric-data" / "cache" / "kubric_fresh").mkdir(parents=True)
    result = sandbox.run("extract", "kubric_fresh")
    assert result.returncode == 2 and "before the first extraction" in result.stderr
    assert not any(c.startswith("extract") for c in sandbox.commands())


def test_an_interrupted_extraction_resumes_into_its_own_cache(sandbox: FinalSandbox) -> None:
    sandbox.ok("seed")
    sandbox.ok("render", "kubric_fresh")
    sandbox.ok("extract", "kubric_fresh")
    sandbox.ok("extract", "kubric_fresh")
    sandbox.ok("extract", "kubric_synthetic_eval")
    extracts = [c for c in sandbox.commands() if c.startswith("extract")]
    assert extracts == [
        "extract run --dataset kubric_fresh --subset eval",
        "extract run --dataset kubric_fresh --subset eval",
        "extract run --dataset kubric_synthetic --subset eval",
    ]


def test_extract_shards_warm_the_cache_before_the_locked_pass(sandbox: FinalSandbox) -> None:
    sandbox.ok("seed")
    sandbox.ok("render", "kubric_fresh")
    env = {"PHYSVERIFY_FINAL_EXTRACT_LIMIT": "2", "PHYSVERIFY_FINAL_EXTRACT_SHARDS": "2"}
    sandbox.ok("extract", "kubric_fresh", **env)
    pv = [line[3:] for line in sandbox.lines() if line.startswith("pv ") and "extract" in line]
    shard = "--profile precise --device cpu --set runtime.threads=16 --no-lock extract run "
    clips = "--dataset kubric_fresh --subset eval --clips kubric_fresh-"
    assert sorted(pv[:2]) == [f"{shard}{clips}1", f"{shard}{clips}2"]
    assert pv[2] == (
        "--profile precise --device cpu --set runtime.threads=32 extract run --dataset "
        "kubric_fresh --subset eval --clips kubric_fresh-1 --clips kubric_fresh-2"
    )


def _ready(sandbox: FinalSandbox, dataset: str = "kubric_fresh") -> None:
    sandbox.ok("seed")
    sandbox.ok("render", dataset)
    sandbox.ok("extract", dataset)


def test_a_set_is_looked_at_once(sandbox: FinalSandbox) -> None:
    """A second look and a retry after a written evaluation are both refused."""
    _ready(sandbox)
    sandbox.ok("look", "kubric_fresh")
    for args in (("look", "kubric_fresh"), ("look", "kubric_fresh", "--retry")):
        result = sandbox.run(*args)
        assert result.returncode == 2 and "one look per set" in result.stderr, result.stderr
    assert len(sandbox.looks()) == 1
    assert len(sandbox.ledger_lines()) == 1


def test_one_retry_follows_a_look_that_wrote_no_evaluation(sandbox: FinalSandbox) -> None:
    _ready(sandbox)
    failed = sandbox.run("look", "kubric_fresh", EVAL_FAIL_ONCE=str(sandbox.root / "failed"))
    assert failed.returncode == 3
    assert sandbox.run("look", "kubric_fresh").returncode == 2
    retried = sandbox.ok("look", "kubric_fresh", "--retry")
    assert "evaluation" in retried.stdout
    assert sandbox.envs()[-1][1:3] == ["kubric_fresh", "1"]
    assert sandbox.run("look", "kubric_fresh", "--retry").returncode == 2
    assert len(sandbox.looks()) == 2 and len(sandbox.ledger_lines()) == 2
    assert "retry look at kubric_fresh" in sandbox.summary("look-kubric_fresh")


def test_a_render_after_a_look_is_refused(sandbox: FinalSandbox) -> None:
    """No new render of a set that has been looked at."""
    _ready(sandbox)
    sandbox.ok("look", "kubric_fresh")
    result = sandbox.run("render", "kubric_fresh")
    assert result.returncode == 2 and "no render after a look" in result.stderr
    assert len([line for line in sandbox.lines() if line.startswith("kubric ")]) == 1


def test_a_missing_ledger_stops_the_look(sandbox: FinalSandbox) -> None:
    _ready(sandbox)
    sandbox.ledger.unlink()
    result = sandbox.run("look", "kubric_fresh")
    assert result.returncode == 2 and "fail closed" in result.stderr
    assert sandbox.looks() == []


def test_a_look_needs_its_extraction(sandbox: FinalSandbox) -> None:
    sandbox.ok("seed")
    sandbox.ok("render", "kubric_fresh")
    result = sandbox.run("look", "kubric_fresh_blind")
    assert (
        result.returncode == 2 and "extract kubric_fresh_blind has not completed" in result.stderr
    )
    sandbox.ok("extract", "kubric_fresh_blind")
    sandbox.ok("look", "kubric_fresh_blind")
    assert sandbox.looks() == [
        _look(sandbox.final, "--dataset", "kubric_fresh_blind", "--subset", "eval", "--no-real")
    ]


def test_the_new_real_control_is_fetched_once_and_looked_at_together(sandbox: FinalSandbox) -> None:
    """The downloaded files are fixed by their manifest; one look covers both sets."""
    sandbox.ok("seed")
    for name in ("physics_iq_views", "objectron_static"):
        sandbox.ok("fetch", name)
    assert sandbox.run("look", "real_new").returncode == 2
    for name in ("physics_iq_views", "objectron_static"):
        sandbox.ok("extract", name)
    refetch = sandbox.run("fetch", "objectron_static")
    assert refetch.returncode == 2 and "downloaded files are fixed" in refetch.stderr
    sandbox.ok("look", "real_new")
    assert sandbox.looks() == [_look(sandbox.final, *REAL_NEW)]
    assert sandbox.envs()[-1][1] == "physics_iq_views,objectron_static"
    fetches = [line for line in sandbox.lines() if line.startswith("fetch ")]
    assert fetches == [
        "fetch physics_iq_views seed " + str(sandbox.seed),
        "fetch objectron_static seed " + str(sandbox.seed),
    ]
    manifest = sandbox.final / "run-final" / "fetch-objectron_static.sha256"
    assert manifest.read_text("utf-8").count("\n") == 2


def test_changed_downloads_are_refused(sandbox: FinalSandbox) -> None:
    sandbox.ok("seed")
    sandbox.ok("fetch", "physics_iq_views")
    raw = sandbox.root / "kubric-data" / "raw" / "physics_iq_views" / "a.mp4"
    raw.write_text("changed", encoding="utf-8")
    result = sandbox.run("extract", "physics_iq_views")
    assert result.returncode == 2 and "not those of" in result.stderr
    assert not any(c.startswith("extract") for c in sandbox.commands())


def test_a_skipped_real_dataset_fails_the_look(sandbox: FinalSandbox) -> None:
    sandbox.ok("seed")
    sandbox.ok("extract", "kubric_synthetic_eval")
    result = sandbox.run("look", "kubric_synthetic_eval", SKIPPED="1")
    assert result.returncode == 2 and "skipped" in result.stderr
    assert sandbox.looks() == [
        _look(sandbox.final, "--dataset", "kubric_synthetic", "--subset", "eval", "--real")
    ]
    assert sandbox.envs()[-1][1] == "kubric_synthetic:eval"


@pytest.mark.parametrize(
    ("setup", "env", "message"),
    [
        ("none", {}, "look kubric_fresh has not completed"),
        ("look", {"CKV_SHA": "b" * 40}, "is not at the v1 commit"),
        ("look", {"CKV_DIRTY": " M configs/checks.toml\n"}, "uncommitted"),
        ("look", {"META": "1"}, "recomputed stages"),
        ("base", {}, "not the freeze's v1 calibration"),
        ("configs", {}, "beyond checks.toml"),
    ],
)
def test_the_v1_arm_is_refused_unless_it_repeats_the_look_with_the_base_state(
    sandbox: FinalSandbox, setup: str, env: dict[str, str], message: str
) -> None:
    _ready(sandbox)
    if setup != "none":
        sandbox.ok("look", "kubric_fresh")
    if setup == "base":
        (sandbox.base / "calibration.json").write_text("another base", encoding="utf-8")
    if setup == "configs":
        extra = sandbox.final / "v1arm" / "configs"
        extra.mkdir(parents=True)
        (extra / "sealed.toml").write_text("base sealed\n", encoding="utf-8")
    result = sandbox.run("v1arm", "kubric_fresh", **env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    if setup != "look" or "META" not in env:
        assert not any(q.startswith("overlay_cli.py") for q in sandbox.queries())


DRY_SEED = 999
UNSEALED = {
    "PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS": '["center"]',
    "PHYSVERIFY__OBJECTRON_STATIC__SPLIT": "train",
}


def _dry_run(sandbox: FinalSandbox) -> tuple[Path, Path, dict[str, str]]:
    """A rehearsal freeze of seed ``DRY_SEED`` under scratch; the real freeze stays in place."""
    scratch = sandbox.root / "scratch"
    final = scratch / "final"
    sandbox.freeze(final, seed=DRY_SEED)
    ledger = scratch / "ledger.jsonl"
    ledger.touch()
    env = {
        "PHYSVERIFY_FINAL_DRYRUN": "1",
        "PHYSVERIFY_FINAL_DIR": str(final),
        "PHYSVERIFY_FINAL_DATA": str(scratch / "data"),
        "PHYSVERIFY_FINAL_SEED": "7",
        "PHYSVERIFY__SEALED__LEDGER": str(ledger),
        "PHYSVERIFY__KUBRIC_FRESH__SCENES_PER_FAMILY": "1",
    }
    return final, ledger, env


def test_a_dry_run_writes_only_under_scratch(sandbox: FinalSandbox) -> None:
    """Its own freeze, data, cache and ledger; the seed is not the freeze's."""
    final, ledger, env = _dry_run(sandbox)
    scratch = final.parent
    for args in (("seed",), ("render", "kubric_fresh"), ("extract", "kubric_fresh")):
        sandbox.ok(*args, **env)
    sandbox.ok("look", "kubric_fresh", **env)
    assert sandbox.ledger_lines() == [] and len(sandbox.ledger_lines(ledger)) == 1
    assert "not checked against the formula" in sandbox.summary("seed", final)
    assert all(e[3].startswith(str(scratch)) for e in sandbox.envs())
    outside = sandbox.run("seed", **{**env, "PHYSVERIFY_FINAL_DATA": str(sandbox.root / "data")})
    assert outside.returncode == 2 and "writes only under" in outside.stderr
    real = {k: v for k, v in env.items() if not k.startswith("PHYSVERIFY__")}
    refused = sandbox.run("seed", **{**real, "PHYSVERIFY_FINAL_DRYRUN": "0"})
    assert refused.returncode == 2 and "only a dry run writes there" in refused.stderr


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (("seed",), {"PHYSVERIFY_FINAL_SEED": str(DRY_SEED)}, "with the freeze's seed"),
        (("seed",), {"PHYSVERIFY_FINAL_SEED": "real"}, "with the seed of"),
        (("extract", "kubric_synthetic_eval"), {}, "never touches kubric_synthetic_eval"),
        (("fetch", "objectron_static"), {}, "PHYSVERIFY__OBJECTRON_STATIC__SPLIT='train'"),
        (
            ("fetch", "objectron_static"),
            {"PHYSVERIFY__OBJECTRON_STATIC__SPLIT": "test"},
            "PHYSVERIFY__OBJECTRON_STATIC__SPLIT='train'",
        ),
        (("fetch", "physics_iq_views"), {}, """PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS='["center"]'"""),
        (
            ("extract", "physics_iq_views"),
            {"PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS": '["left", "right"]'},
            "PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS",
        ),
        (
            ("look", "real_new"),
            {"PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS": '["center"]'},
            "PHYSVERIFY__OBJECTRON_STATIC__SPLIT",
        ),
        (
            ("look", "real_new"),
            {"PHYSVERIFY__OBJECTRON_STATIC__SPLIT": "train"},
            "PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS",
        ),
    ],
)
def test_a_dry_run_never_reads_a_sealed_population(
    sandbox: FinalSandbox, args: tuple[str, ...], env: dict[str, str], message: str
) -> None:
    """Audit B1, M1: no frozen seed, no kubric_synthetic eval, no left/right view, no test split."""
    _, ledger, dry = _dry_run(sandbox)
    if env.get("PHYSVERIFY_FINAL_SEED") == "real":
        env = {"PHYSVERIFY_FINAL_SEED": str(sandbox.seed)}
    result = sandbox.run(*args, **{**dry, **env})
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == []
    assert sandbox.ledger_lines(ledger) == []


def test_a_dry_run_of_the_new_real_control_reads_center_and_train(sandbox: FinalSandbox) -> None:
    _, ledger, dry = _dry_run(sandbox)
    env = {**dry, **UNSEALED}
    sandbox.ok("seed", **env)
    for name in ("physics_iq_views", "objectron_static"):
        sandbox.ok("fetch", name, **env)
        sandbox.ok("extract", name, **env)
    sandbox.ok("look", "real_new", **env)
    assert len(sandbox.ledger_lines(ledger)) == 1 and sandbox.ledger_lines() == []


def test_a_limited_extraction_is_not_the_extraction_a_look_needs(sandbox: FinalSandbox) -> None:
    """Audit M2: the pace measurement is recorded as extract-limited."""
    sandbox.ok("seed")
    sandbox.ok("render", "kubric_fresh")
    sandbox.ok("extract", "kubric_fresh", PHYSVERIFY_FINAL_EXTRACT_LIMIT="2")
    steps = (sandbox.final / "run-final" / "steps.tsv").read_text("utf-8").splitlines()
    assert [line.split("\t")[:3] for line in steps][-1] == ["extract-limited", "kubric_fresh", "0"]
    assert (sandbox.final / "run-final" / "summary-extract-limited-kubric_fresh.md").is_file()
    result = sandbox.run("look", "kubric_fresh")
    assert result.returncode == 2 and "extract kubric_fresh has not completed" in result.stderr
    assert sandbox.looks() == []
    sandbox.ok("extract", "kubric_fresh")
    sandbox.ok("look", "kubric_fresh")
    assert len(sandbox.looks()) == 1


def test_a_render_without_a_planned_clip_count_stops(sandbox: FinalSandbox) -> None:
    """The exported share is never divided by an empty or zero plan."""
    sandbox.ok("seed")
    result = sandbox.run("render", "kubric_fresh", PLAN_CLIPS="0")
    assert result.returncode == 2 and "no planned clip count" in result.stderr


def test_an_unchanged_checks_toml_is_not_a_foreign_config_difference(sandbox: FinalSandbox) -> None:
    """An empty configs diff (the v1 commit's checks.toml is the freeze's) passes."""
    (sandbox.ckv / "configs" / "checks.toml").write_text("frozen checks\n", encoding="utf-8")
    _ready(sandbox)
    sandbox.ok("look", "kubric_fresh")
    sandbox.ok("v1arm", "kubric_fresh")
    assert (sandbox.final / "v1arm" / "configs.diff").read_text("utf-8") == "\n"


def test_the_v1_arm_flattens_the_nested_configs(sandbox: FinalSandbox) -> None:
    """The v1 checkout reads one flat directory; only checks.toml differs from the freeze."""
    _ready(sandbox)
    sandbox.ok("look", "kubric_fresh")
    sandbox.ok("v1arm", "kubric_fresh")
    arm = sandbox.final / "v1arm"
    assert sorted(p.name for p in (arm / "configs").iterdir()) == [
        "checks.toml",
        "default.toml",
        "fusion.toml",
    ]
    frozen = sandbox.repo / "configs" / "verifier" / "checks.toml"
    assert (arm / "configs.diff").read_text("utf-8") == (
        f"diff -r {frozen} {arm / 'configs' / 'checks.toml'}\n"
        "1c1\n< frozen checks\n---\n> v1 checks\n"
    )


def test_two_config_files_sharing_a_name_cannot_be_flattened(sandbox: FinalSandbox) -> None:
    (sandbox.repo / "configs" / "verifier" / "fusion.toml").write_text("other\n", encoding="utf-8")
    _ready(sandbox)
    sandbox.ok("look", "kubric_fresh")
    result = sandbox.run("v1arm", "kubric_fresh")
    assert result.returncode == 2 and "holds two files named fusion.toml" in result.stderr
    assert not (sandbox.final / "v1arm" / "configs").exists()


@pytest.mark.parametrize(
    ("place", "message"),
    [("repo", "beyond checks.toml: no extra.toml"), ("arm", "beyond checks.toml: extra files")],
)
def test_a_config_file_on_one_side_only_stops_the_v1_arm(
    sandbox: FinalSandbox, place: str, message: str
) -> None:
    """A file the other side lacks (an untracked local.toml, say) is never ignored."""
    _ready(sandbox)
    sandbox.ok("look", "kubric_fresh")
    arm = sandbox.final / "v1arm" / "configs"
    arm.mkdir(parents=True)
    for name, text in REPO_CONFIG_FILES.items():
        (arm / Path(name).name).write_text(text, encoding="utf-8")
    extra = sandbox.repo / "configs" / "studies" if place == "repo" else arm
    (extra / "extra.toml").write_text("extra\n", encoding="utf-8")
    result = sandbox.run("v1arm", "kubric_fresh")
    assert result.returncode == 2 and message in result.stderr, result.stderr


def test_a_config_file_other_than_checks_that_differs_stops_the_v1_arm(
    sandbox: FinalSandbox,
) -> None:
    _ready(sandbox)
    sandbox.ok("look", "kubric_fresh")
    arm = sandbox.final / "v1arm" / "configs"
    arm.mkdir(parents=True)
    for name, text in REPO_CONFIG_FILES.items():
        (arm / Path(name).name).write_text(text, encoding="utf-8")
    (arm / "fusion.toml").write_text("edited fusion\n", encoding="utf-8")
    result = sandbox.run("v1arm", "kubric_fresh")
    assert result.returncode == 2, result.stderr
    assert "beyond checks.toml: fusion.toml" in result.stderr, result.stderr


def test_a_rerun_of_the_v1_arm_needs_the_same_configs_diff(sandbox: FinalSandbox) -> None:
    _ready(sandbox)
    sandbox.ok("look", "kubric_fresh")
    sandbox.ok("v1arm", "kubric_fresh")
    (sandbox.final / "v1arm" / "configs" / "checks.toml").write_text("edited\n", encoding="utf-8")
    result = sandbox.run("v1arm", "kubric_fresh")
    assert result.returncode == 2 and "configs changed since" in result.stderr, result.stderr
