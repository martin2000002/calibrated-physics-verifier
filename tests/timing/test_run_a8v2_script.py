"""scripts/h200/run_a8v2.sh against a fake interpreter, ``ps``, ``git`` and migration script."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from check_rows import UNCACHED, write_check_rows

REPO = Path(__file__).resolve().parents[2]
H200 = REPO / "scripts" / "h200"
SCRIPT = H200 / "run_a8v2.sh"
TOOLS = (
    "basename", "bash", "cat", "cmp", "cp", "dirname", "env", "grep", "head", "mkdir", "mv",
    "nice", "nproc", "realpath", "sed", "sort", "tee", "touch", "tr",
)  # fmt: skip
PREFIX = re.compile(r"^--profile precise --device cpu --set runtime\.threads=\d+ ")
# The query helper runs for real on run-errors (it reads the checks.jsonl verify wrote) unless the
# run holds the clean rows and REAL_QUERY is unset: its import costs seconds per call.
FAKE_PYTHON = r"""#!/bin/bash
if [[ "$1" != -c ]]; then
    printf 'query %s\n' "${*:2}" >> "$CALLS"
    case "$*" in
      *"run-errors "*)
        [[ -z "${REAL_QUERY:-}" ]] && cmp -s "$CLEAN_CHECKS" "${!#}/checks.jsonl" && exit 0
        PYTHONPATH="$REAL_SRC" exec "$REAL_PYTHON" "$@" ;;
      *"clips pisa_real full") printf '%s\n' ${PISA_SELECTED:-p1 p2} ;;
      *"clips kubric_synthetic full --split calib") printf 'k1\nk2\nk3\n' ;;
      *"clips kubric_synthetic full") printf 'k1\nk2\nk3\nk4\nk5\n' ;;
      *"clips physics_iq full") echo q1 ;;
      *"rerendered "*"--missing-ok") echo k1 ;;
      *"rerendered "*) echo k1; exit "${MIGRATED_EXIT:-0}" ;;
      *"evaluation "*) echo "## H1 primary"; exit "${EVAL_QUERY_EXIT:-0}" ;;
      *"gravity "*) echo "| lift | 3 |" ;;
    esac
    exit 0
fi
shift 2
printf 'pv %s\n' "$*" >> "$CALLS"
new_run() {
    local n
    n=$(( $(cat "$DATA/counter" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$DATA/counter"
    RUN="$DATA/runs/r$n"
    mkdir -p "$RUN/diagnosis"
    if [[ -n "${HASH:-}" ]]; then echo "h$n"; else echo same; fi > "$RUN/checks_hash.txt"
    echo "run: $RUN"
}
case "$*" in
  *" doctor")
    echo "paths.cache_dir: ${CACHE:-$DATA/cache} (env)"
    echo "paths.error_profile: ${PROFILE_FILE:-$DATA/calibration/stage_errors.precise.json}"
    echo "paths.synthetic_dir: $DATA/synthetic"
    echo "paths.data_dir: $DATA"
    echo "heavy_lock: $PHYSVERIFY__RUNTIME__HEAVY_LOCK (free)" ;;
  *" verify "*)
    new_run
    if [[ -n "${NOT_CACHED:-}" ]]; then cp "$UNCACHED_CHECKS" "$RUN/checks.jsonl"
    else cp "$CLEAN_CHECKS" "$RUN/checks.jsonl"; fi ;;
  *" extract run "*) new_run; exit "${EXTRACT_EXIT:-0}" ;;
  *"--no-lock diagnose stages "*)
    [[ "$* " != *" --clips ${FAILED_SHARD_CLIP:-none} "* ]] || exit "${SHARD_EXIT:-1}" ;;
  *" diagnose stages "*) new_run ;;
  *" calibrate evaluate "*)
    new_run
    echo '{}' > "$RUN/evaluation.json"
    [[ -z "${SKIP:-}" ]] || echo "real dataset pisa_real skipped: missing"
    echo "evaluation: $RUN/evaluation.json" ;;
  *" calibrate fit "*) exit "${FIT_EXIT:-0}" ;;
  *" exclusions r2-clips "*)
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == --out ]]; then printf 'k2\nk3\n' > "$2"; fi
        shift
    done ;;
  *" exclusions show") echo "record: /x/record.json sha256 abc123" ;;
esac
exit 0
"""
FAKE_MIGRATE = """#!/bin/bash
printf 'migrate %s %s\\n' "$PHYSVERIFY_MIGRATE_NAME" "${PHYSVERIFY_LOCK_HELD-unset}" >> "$CALLS"
"""
FAKE_PS = '#!/bin/bash\nprintf "%s\\n" "${PS_OUT:-  1 init}"\n'
FAKE_GIT = '#!/bin/bash\n[[ "$*" != *" diff "* ]] || exit "${DIRTY:-0}"\necho c0ffee1234567\n'
STEP_NAMES = (
    "migrate", "extract-synthetic", "stages", "relift", "stage-sigma", "fit", "evaluate",
    "abstain", "r2-arms", "decide", "size-eval", "gravity",
)  # fmt: skip


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.data = self.root / "kubric-data"
        self.out = self.root / "a8v2"
        self.calls = self.root / "calls.log"
        scripts = self.root / "repo" / "scripts" / "h200"
        scripts.mkdir(parents=True)
        self.script = scripts / SCRIPT.name
        shutil.copy2(SCRIPT, self.script)
        shutil.copy2(H200 / "_common.sh", scripts / "_common.sh")
        shutil.copy2(H200 / "a8v2_query.py", scripts / "a8v2_query.py")
        _executable(scripts / "migrate_velocity.sh", FAKE_MIGRATE)
        tools, fakes = self.root / "tools", self.root / "fakes"
        tools.mkdir()
        fakes.mkdir()
        for tool in TOOLS:
            found = shutil.which(tool)
            assert found is not None, tool
            (tools / tool).symlink_to(found)
        _executable(fakes / "python", FAKE_PYTHON)
        _executable(fakes / "ps", FAKE_PS)
        _executable(fakes / "git", FAKE_GIT)
        shards = self.root / "logs" / "shards"
        shards.mkdir(parents=True)
        (shards / "kubric_full.ids").write_text("k1 k2 k3\nk4\n")
        (shards / "physics_iq_full.ids").write_text("q1\n")
        (shards / "pisa_real_full.ids").write_text("p1 p2\n")
        self.data.mkdir()
        self.env = {
            "PATH": f"{fakes}:{tools}",
            "REAL_PYTHON": sys.executable,
            "REAL_SRC": str(REPO / "src"),
            "CLEAN_CHECKS": str(write_check_rows(self.root / "clean.jsonl")),
            "UNCACHED_CHECKS": str(write_check_rows(self.root / "uncached.jsonl", error=UNCACHED)),
            "CALLS": str(self.calls),
            "DATA": str(self.data),
            "PHYSVERIFY_H200_ROOT": str(self.root),
            "PHYSVERIFY_H200_ENV": str(self.root / "no-env.sh"),
            "PHYSVERIFY_A8V2_PYTHON": str(fakes / "python"),
            "PHYSVERIFY_A8V2_EXPECT_COMMIT": "c0ffee1",
            "PHYSVERIFY_A8V2_CACHE_DIR": str(self.data / "cache"),
            "PHYSVERIFY_LOCK_HELD": "1",
        }

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.root / "tools" / "bash"), str(self.script), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def lines(self) -> list[str]:
        return self.calls.read_text(encoding="utf-8").splitlines() if self.calls.exists() else []

    def commands(self) -> list[str]:
        """The pv calls without their common prefix, plus the migration."""
        return [
            PREFIX.sub("", line.removeprefix("pv "))
            for line in self.lines()
            if line.startswith(("pv ", "migrate "))
        ]

    def steps(self) -> list[list[str]]:
        text = (self.out / "steps.tsv").read_text(encoding="utf-8")
        return [line.split("\t") for line in text.splitlines()]


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def test_script_parses_and_is_executable() -> None:
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, "-n", str(SCRIPT)], check=True)
    assert os.access(SCRIPT, os.X_OK)


def test_a_running_extraction_stops_it_before_any_call(sandbox: Sandbox) -> None:
    result = sandbox.run(PS_OUT="4242 python -c from physverify.cli.main import main extract run")
    assert result.returncode == 2
    assert "still running" in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_A8V2_EXPECT_COMMIT": ""}, "PHYSVERIFY_A8V2_EXPECT_COMMIT"),
        ({"PHYSVERIFY_A8V2_EXPECT_COMMIT": "deadbeef"}, "expected deadbeef"),
        ({"PHYSVERIFY_A8V2_CACHE_DIR": ""}, "PHYSVERIFY_A8V2_CACHE_DIR"),
    ],
)
def test_commit_and_cache_are_required(sandbox: Sandbox, env: dict[str, str], message: str) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2
    assert message in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"CACHE": "/elsewhere/cache"}, "not the shards'"),
        ({"PROFILE_FILE": "/c/stage_errors.recalibrated.json"}, "not stage_errors.precise.json"),
    ],
)
def test_a_cache_or_profile_other_than_the_shards_stops_after_doctor(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2
    assert message in result.stderr
    assert sandbox.commands() == ["doctor"]


def test_r2_arms_with_different_check_parameters_stop_before_decide(sandbox: Sandbox) -> None:
    result = sandbox.run("8", HASH="1")
    assert result.returncode == 2
    assert "different check parameters" in result.stderr
    assert sandbox.steps()[-1][:3] == ["8", "r2-arms", "2"]


@pytest.mark.parametrize("start", ["x", "12", "-1", "01"])
def test_a_bad_start_step_is_a_usage_error(sandbox: Sandbox, start: str) -> None:
    result = sandbox.run(start)
    assert result.returncode == 2
    assert sandbox.lines() == []


def test_the_full_run_preflights_then_runs_every_step_in_order(sandbox: Sandbox) -> None:
    result = sandbox.run(REAL_QUERY="1")
    assert result.returncode == 0, result.stderr
    assert all(PREFIX.match(line[3:]) for line in sandbox.lines() if line.startswith("pv "))
    ks = "--dataset kubric_synthetic --subset full"
    runs = sandbox.data / "runs"
    assert sandbox.commands() == [
        "doctor",
        "verify --dataset kubric_synthetic --level perceived --subset full --clips k2",
        "verify --dataset physics_iq --level perceived --subset full --clips q1",
        "verify --dataset pisa_real --level perceived --subset full --clips p1",
        f"diagnose stages {ks} --clips k2",
        "migrate pair-collision unset",
        f"extract run {ks}",
        f"diagnose stages {ks} --write-profile",
        f"extract run {ks}",
        "extract run --dataset physics_iq --subset full",
        "extract run --dataset pisa_real --subset full",
        f"calibrate stage-sigma {ks}",
        f"calibrate fit {ks} --level both",
        f"calibrate evaluate {ks} --real",
        "abstain fit",
        "abstain curve",
        f"exclusions r2-clips --out {sandbox.out / 'r2_clips.txt'}",
        f"verify {ks} --level perceived --policy raw --clips k2 --clips k3",
        f"--set extraction.plan.detect=grounding_dino_tiny extract run {ks} --clips k2 --clips k3",
        f"--set extraction.plan.detect=grounding_dino_tiny verify {ks} --level perceived "
        "--policy raw --clips k2 --clips k3",
        f"exclusions decide --evaluation {runs / 'r10' / 'evaluation.json'} "
        "--r2-base-run r11 --r2-tiny-run r13",
        "exclusions show",
        "size eval",
    ]
    assert [row[1] for row in sandbox.steps()] == ["preflight", *STEP_NAMES]
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "sha256 abc123" in summary
    assert "## H1 primary" in summary and "| lift | 3 |" in summary
    assert "steps 2 and 4 each take about" in summary
    assert "RECORD_SHA=abc123" in (sandbox.out / "state.env").read_text(encoding="utf-8")


def test_a_resumed_decide_reads_the_saved_state_and_skips_the_preflight(sandbox: Sandbox) -> None:
    assert sandbox.run().returncode == 0
    sandbox.calls.unlink()
    result = sandbox.run("9", PHYSVERIFY_A8V2_DECIDE_REPLACE="1")
    assert result.returncode == 0, result.stderr
    assert sandbox.commands()[1].endswith("--r2-base-run r11 --r2-tiny-run r13 --replace")
    assert not any(line.startswith("query clips") for line in sandbox.lines())


def test_decide_without_saved_runs_asks_to_rerun_them(sandbox: Sandbox) -> None:
    result = sandbox.run("9")
    assert result.returncode == 2
    assert "rerun from 6" in result.stderr
    assert sandbox.steps()[-1][:3] == ["9", "decide", "2"]


def test_resuming_after_the_migration_does_not_need_it(sandbox: Sandbox) -> None:
    result = sandbox.run("1")
    assert result.returncode == 0, result.stderr
    assert not any(c.startswith("migrate") for c in sandbox.commands())


def test_an_uncached_preflight_clip_stops_before_the_migration(sandbox: Sandbox) -> None:
    """verify exits 0 on an uncached clip; only its checks.jsonl carries the error."""
    result = sandbox.run(NOT_CACHED="1")
    assert result.returncode == 2
    assert "is not cached" in result.stderr and "heavy stage is not cached" in result.stderr
    assert sandbox.steps()[-1][:3] == ["pre", "preflight", "2"]
    assert not any(c.startswith("migrate") for c in sandbox.commands())


def test_uncommitted_changes_stop_it_before_any_call(sandbox: Sandbox) -> None:
    result = sandbox.run(DIRTY="1")
    assert result.returncode == 2
    assert "uncommitted changes" in result.stderr
    assert sandbox.lines() == []


def _audit(sandbox: Sandbox) -> Path:
    audit = sandbox.data / "runs" / "migrate-pair-collision-v1-precise" / "audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text("{}", encoding="utf-8")
    return audit


def test_a_completed_migration_is_not_rerun(sandbox: Sandbox) -> None:
    audit = _audit(sandbox)
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert not any(c.startswith("migrate") for c in sandbox.commands())
    assert "already complete" in (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert str(audit) in (sandbox.out / "summary.md").read_text(encoding="utf-8")


def test_an_incomplete_manifest_reruns_the_migration(sandbox: Sandbox) -> None:
    _audit(sandbox)
    result = sandbox.run(MIGRATED_EXIT="2")
    assert result.returncode == 2
    assert "migrate pair-collision unset" in sandbox.commands()
    assert sandbox.steps()[-1][:3] == ["1", "extract-synthetic", "2"]


def _timed(sandbox: Sandbox) -> list[str]:
    return [c for c in sandbox.commands() if c.startswith("diagnose stages") and "--clips" in c]


def test_the_estimate_times_a_clip_once_and_never_after_step_2(sandbox: Sandbox) -> None:
    ks = "--dataset kubric_synthetic --subset full"
    assert sandbox.run("3", FIT_EXIT="1").returncode == 1
    assert _timed(sandbox) == []
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "estimate: none, step 2 has already begun" in summary
    assert sandbox.run("1", FIT_EXIT="1").returncode == 1
    assert _timed(sandbox) == [f"diagnose stages {ks} --clips k2"]
    assert "model loading included" in (sandbox.out / "summary.md").read_text(encoding="utf-8")
    sandbox.calls.unlink()
    assert sandbox.run("1").returncode == 0
    assert _timed(sandbox) == []
    assert "step 2 has already begun" in (sandbox.out / "summary.md").read_text(encoding="utf-8")


def test_an_interrupted_estimate_does_not_time_the_same_clip_again(sandbox: Sandbox) -> None:
    ks = "--dataset kubric_synthetic --subset full"
    (sandbox.out).mkdir()
    (sandbox.out / "state.env").write_text("TIMED_CLIPS=k2\n", encoding="utf-8")
    assert sandbox.run("1", FIT_EXIT="1").returncode == 1
    assert _timed(sandbox) == [f"diagnose stages {ks} --clips k3"]
    (sandbox.out / "state.env").write_text("TIMED_CLIPS='k2 k3'\n", encoding="utf-8")
    sandbox.calls.unlink()
    assert sandbox.run("1", FIT_EXIT="1").returncode == 1
    assert _timed(sandbox) == []
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "every extracted calib clip has been timed before" in summary


def test_failed_clips_of_extract_are_tolerated_and_noted(sandbox: Sandbox) -> None:
    result = sandbox.run("1", EXTRACT_EXIT="1")
    assert result.returncode == 0, result.stderr
    assert "some clips failed" in (sandbox.out / "summary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("code", ["1", "75"])
def test_a_failed_fit_stops_with_its_code(sandbox: Sandbox, code: str) -> None:
    result = sandbox.run("5", FIT_EXIT=code)
    assert result.returncode == int(code)
    assert sandbox.steps()[-1][:3] == ["5", "fit", code]
    assert sandbox.commands()[-1].startswith("calibrate fit")


def test_a_skipped_real_dataset_stops_the_evaluation(sandbox: Sandbox) -> None:
    result = sandbox.run("6", SKIP="1")
    assert result.returncode == 2
    assert "skipped" in result.stderr
    assert sandbox.commands()[-1].startswith("calibrate evaluate")


def test_an_evaluation_failing_its_checks_stops_the_run(sandbox: Sandbox) -> None:
    result = sandbox.run("6", EVAL_QUERY_EXIT="2")
    assert result.returncode == 2
    assert sandbox.steps()[-1][:3] == ["6", "evaluate", "2"]


def test_a_pisa_selection_other_than_the_extracted_one_stops_at_startup(sandbox: Sandbox) -> None:
    result = sandbox.run(PISA_SELECTED="p1")
    assert result.returncode == 2
    assert "set PHYSVERIFY_A8V2_PISA_MAX_CLIPS" in result.stderr
    assert sandbox.commands() == ["doctor"]
    outside = sandbox.run(PISA_SELECTED="p1 p9", PHYSVERIFY_A8V2_PISA_MAX_CLIPS="2")
    assert outside.returncode == 2
    assert "were not extracted" in outside.stderr


def test_a_pisa_subset_applies_to_every_call(sandbox: Sandbox) -> None:
    result = sandbox.run("2", PISA_SELECTED="p2", PHYSVERIFY_A8V2_PISA_MAX_CLIPS="1")
    assert result.returncode == 0, result.stderr
    calls = [line for line in sandbox.lines() if line.startswith(("pv ", "query "))]
    assert calls and all("--set pisa_real.max_clips=1" in line for line in calls)


def _shards(sandbox: Sandbox) -> list[str]:
    return sorted(c for c in sandbox.commands() if c.startswith("--no-lock"))


def test_diag_shards_warm_the_cache_before_the_profile_call(sandbox: Sandbox) -> None:
    ks = "--dataset kubric_synthetic --subset full"
    result = sandbox.run("2", PHYSVERIFY_A8V2_DIAG_SHARDS="2", FIT_EXIT="1")
    assert result.returncode == 1
    assert _shards(sandbox) == [
        f"--no-lock diagnose stages {ks} --clips k1 --clips k3 --clips k5",
        f"--no-lock diagnose stages {ks} --clips k2 --clips k4",
    ]
    shard_lines = [line for line in sandbox.lines() if "--no-lock" in line]
    assert all("--set runtime.threads=16 " in line for line in shard_lines)
    assert not any("--write-profile" in line for line in shard_lines)
    commands = sandbox.commands()
    profile = commands.index(f"diagnose stages {ks} --write-profile")
    assert all(commands.index(shard) < profile for shard in _shards(sandbox))
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "step 2 shard 1 (16 threads): exit 0" in summary


def test_diag_shards_threads_can_be_set(sandbox: Sandbox) -> None:
    env = {"PHYSVERIFY_A8V2_DIAG_SHARDS": "8", "PHYSVERIFY_A8V2_DIAG_THREADS": "3"}
    assert sandbox.run("2", FIT_EXIT="1", **env).returncode == 1
    assert len(_shards(sandbox)) == 5
    shard_lines = [line for line in sandbox.lines() if "--no-lock" in line]
    assert all("--set runtime.threads=3 " in line for line in shard_lines)


def test_a_diag_shard_with_failed_clips_is_tolerated(sandbox: Sandbox) -> None:
    env = {"PHYSVERIFY_A8V2_DIAG_SHARDS": "2", "FAILED_SHARD_CLIP": "k4"}
    assert sandbox.run("2", FIT_EXIT="1", **env).returncode == 1
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "step 2 shard 1 (16 threads): exit 1" in summary
    assert any("--write-profile" in c for c in sandbox.commands())


@pytest.mark.parametrize("code", ["75", "3"])
def test_a_failed_diag_shard_stops_step_2_before_the_profile(sandbox: Sandbox, code: str) -> None:
    env = {"PHYSVERIFY_A8V2_DIAG_SHARDS": "2", "FAILED_SHARD_CLIP": "k4", "SHARD_EXIT": code}
    result = sandbox.run("2", **env)
    assert result.returncode == 2
    assert "1 of 2 diagnose shards failed" in result.stderr
    assert sandbox.steps()[-1][:3] == ["2", "stages", "2"]
    assert not any("--write-profile" in c for c in sandbox.commands())


@pytest.mark.parametrize(
    "env",
    [{"PHYSVERIFY_A8V2_DIAG_SHARDS": "0"}, {"PHYSVERIFY_A8V2_DIAG_SHARDS": "2",
                                             "PHYSVERIFY_A8V2_DIAG_THREADS": "x"}],
)  # fmt: skip
def test_bad_diag_shard_settings_are_usage_errors(sandbox: Sandbox, env: dict[str, str]) -> None:
    result = sandbox.run("2", **env)
    assert result.returncode == 2
    assert "PHYSVERIFY_A8V2_DIAG_" in result.stderr
    assert sandbox.lines() == []
