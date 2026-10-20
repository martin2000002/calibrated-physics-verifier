"""scripts/h200/run_a8v2.sh against a fake interpreter, ``ps``, ``git`` and migration script."""

from __future__ import annotations

import fcntl
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
    "awk", "basename", "bash", "cat", "cmp", "cp", "dirname", "env", "flock", "grep", "head",
    "mkdir", "mv", "nice", "nproc", "realpath", "sed", "sort", "tee", "touch", "tr",
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
printf 'env %s|%s|%s|%s\n' "${CUDA_VISIBLE_DEVICES-unset}" \
    "${PHYSVERIFY__PATHS__CALIBRATION_DIR-unset}" "${PHYSVERIFY__PATHS__RUNS_DIR-unset}" \
    "$PHYSVERIFY__RUNTIME__HEAVY_LOCK" >> "$CALLS"
CAL="${CAL_DIR:-${PHYSVERIFY__PATHS__CALIBRATION_DIR:-$DATA/calibration}}"
RUNS="${PHYSVERIFY__PATHS__RUNS_DIR:-$DATA/runs}"
new_run() {
    local n
    n=$(( $(cat "$DATA/counter" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$DATA/counter"
    RUN="$RUNS/r$n"
    mkdir -p "$RUN/diagnosis"
    if [[ -n "${HASH:-}" ]]; then echo "h$n"; else echo same; fi > "$RUN/checks_hash.txt"
    echo "run: $RUN"
}
case "$*" in
  *" doctor")
    echo "paths.cache_dir: ${CACHE:-$DATA/cache} (env)"
    echo "device: ${DOCTOR_DEVICE:-$4} (--device)"
    echo "paths.calibration_dir: $CAL"
    echo "paths.runs_dir: $RUNS"
    echo "paths.error_profile: ${PROFILE_FILE:-$CAL/stage_errors.precise.json}"
    echo "paths.thresholds: $CAL/thresholds.json"
    echo "paths.synthetic_dir: $DATA/synthetic"
    echo "paths.data_dir: $DATA"
    echo "heavy_lock: $PHYSVERIFY__RUNTIME__HEAVY_LOCK (free)" ;;
  *" verify "*)
    new_run
    if [[ -n "${NOT_CACHED:-}" ]]; then cp "$UNCACHED_CHECKS" "$RUN/checks.jsonl"
    else cp "$CLEAN_CHECKS" "$RUN/checks.jsonl"; fi ;;
  *" extract run "*)
    [[ -z "${LATE_PS:-}" ]] || echo "$LATE_PS" > "$DATA/ps_late"
    new_run; exit "${EXTRACT_EXIT:-0}" ;;
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
FAKE_PS = """#!/bin/bash
printf "%s\\n" "${PS_OUT:-  1 init}"
cat "$DATA/ps_late" 2>/dev/null || true
"""
FAKE_GIT = '#!/bin/bash\n[[ "$*" != *" diff "* ]] || exit "${DIRTY:-0}"\necho c0ffee1234567\n'
STEP_NAMES = (
    "migrate", "extract-synthetic", "stages", "relift", "stage-sigma", "fit", "evaluate",
    "abstain", "r2-arms", "decide", "size-eval", "gravity",
)  # fmt: skip


SAVES = {
    2: ("STAGES_RUN",), 6: ("EVAL_JSON",), 8: ("R2_BASE_RUN", "R2_TINY_RUN"), 9: ("RECORD_SHA",)
}  # fmt: skip


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

    def rows(self) -> list[list[str]]:
        text = (self.out / "steps.tsv").read_text(encoding="utf-8")
        return [line.split("\t") for line in text.splitlines()]

    def steps(self) -> list[list[str]]:
        """The rows that close a step, without the ones written when it starts."""
        return [row for row in self.rows() if row[2] != "running"]


def _done(sandbox: Sandbox, steps: int, *, saves: int | None = None) -> None:
    """Records steps 0..steps-1 as completed, with the state keys of steps 0..saves-1."""
    sandbox.out.mkdir(exist_ok=True)
    rows = "".join(f"{k}\t{STEP_NAMES[k]}\t0\t1\t-\n" for k in range(steps))
    with (sandbox.out / "steps.tsv").open("a", encoding="utf-8") as tsv:
        tsv.write(rows)
    last = steps if saves is None else saves
    lines = [f"{key}=x{k}\n" for k in range(last) for key in SAVES.get(k, ())]
    with (sandbox.out / "state.env").open("a", encoding="utf-8") as state:
        state.write("".join(lines))


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
    _done(sandbox, 8)
    result = sandbox.run("8", HASH="1")
    assert result.returncode == 2
    assert "different check parameters" in result.stderr
    assert sandbox.steps()[-1][:3] == ["8", "r2-arms", "2"]


@pytest.mark.parametrize(
    "args",
    [("x",), ("12",), ("-1",), ("01",), ("3", "2"), ("0", "12"), ("0", "y"), ("0", "1", "2")],
)
def test_a_bad_step_range_is_a_usage_error(sandbox: Sandbox, args: tuple[str, ...]) -> None:
    result = sandbox.run(*args)
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
    opened = [row[0] for row in sandbox.rows() if row[2] == "running"]
    assert opened == ["pre", *(str(k) for k in range(len(STEP_NAMES)))]
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
    _done(sandbox, 9, saves=6)
    result = sandbox.run("9")
    assert result.returncode == 2
    assert "recorded no EVAL_JSON" in result.stderr and "rerun from 6" in result.stderr
    assert sandbox.commands() == ["doctor"]


def test_resuming_after_the_migration_does_not_need_it(sandbox: Sandbox) -> None:
    _done(sandbox, 1)
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
    _done(sandbox, 3)
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
    _done(sandbox, 1)
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
    _done(sandbox, 1)
    result = sandbox.run("1", EXTRACT_EXIT="1")
    assert result.returncode == 0, result.stderr
    assert "some clips failed" in (sandbox.out / "summary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("code", ["1", "75"])
def test_a_failed_fit_stops_with_its_code(sandbox: Sandbox, code: str) -> None:
    _done(sandbox, 5)
    result = sandbox.run("5", FIT_EXIT=code)
    assert result.returncode == int(code)
    assert sandbox.steps()[-1][:3] == ["5", "fit", code]
    assert sandbox.commands()[-1].startswith("calibrate fit")


def test_a_skipped_real_dataset_stops_the_evaluation(sandbox: Sandbox) -> None:
    _done(sandbox, 6)
    result = sandbox.run("6", SKIP="1")
    assert result.returncode == 2
    assert "skipped" in result.stderr
    assert sandbox.commands()[-1].startswith("calibrate evaluate")


def test_an_evaluation_failing_its_checks_stops_the_run(sandbox: Sandbox) -> None:
    _done(sandbox, 6)
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
    _done(sandbox, 2)
    result = sandbox.run("2", PISA_SELECTED="p2", PHYSVERIFY_A8V2_PISA_MAX_CLIPS="1")
    assert result.returncode == 0, result.stderr
    calls = [line for line in sandbox.lines() if line.startswith(("pv ", "query "))]
    assert calls and all("--set pisa_real.max_clips=1" in line for line in calls)


def _shards(sandbox: Sandbox) -> list[str]:
    return sorted(c for c in sandbox.commands() if c.startswith("--no-lock"))


def test_diag_shards_warm_the_cache_before_the_profile_call(sandbox: Sandbox) -> None:
    ks = "--dataset kubric_synthetic --subset full"
    _done(sandbox, 2)
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
    _done(sandbox, 2)
    assert sandbox.run("2", FIT_EXIT="1", **env).returncode == 1
    assert len(_shards(sandbox)) == 5
    shard_lines = [line for line in sandbox.lines() if "--no-lock" in line]
    assert all("--set runtime.threads=3 " in line for line in shard_lines)


def test_a_diag_shard_with_failed_clips_is_tolerated(sandbox: Sandbox) -> None:
    env = {"PHYSVERIFY_A8V2_DIAG_SHARDS": "2", "FAILED_SHARD_CLIP": "k4"}
    _done(sandbox, 2)
    assert sandbox.run("2", FIT_EXIT="1", **env).returncode == 1
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "step 2 shard 1 (16 threads): exit 1" in summary
    assert any("--write-profile" in c for c in sandbox.commands())


@pytest.mark.parametrize("code", ["75", "3"])
def test_a_failed_diag_shard_stops_step_2_before_the_profile(sandbox: Sandbox, code: str) -> None:
    env = {"PHYSVERIFY_A8V2_DIAG_SHARDS": "2", "FAILED_SHARD_CLIP": "k4", "SHARD_EXIT": code}
    _done(sandbox, 2)
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


KS = "--dataset kubric_synthetic --subset full"
SHARD = "python -c from physverify.cli.main import main; main() --profile precise --device cpu"


def _extracting(dataset: str) -> str:
    return f"4242 {SHARD} extract run --dataset {dataset} --subset full --clips x1"


def _verified(sandbox: Sandbox) -> list[str]:
    return [c.split()[2] for c in sandbox.commands() if c.startswith("verify --dataset")]


def test_a_range_runs_only_its_steps_and_a_later_start_resumes_from_state(
    sandbox: Sandbox,
) -> None:
    first = sandbox.run("0", "2")
    assert first.returncode == 0, first.stderr
    assert "stopped after step 2 as requested" in first.stdout
    assert sandbox.commands() == [
        "doctor",
        "verify --dataset kubric_synthetic --level perceived --subset full --clips k2",
        f"diagnose stages {KS} --clips k2",
        "migrate pair-collision unset",
        f"extract run {KS}",
        f"diagnose stages {KS} --write-profile",
    ]
    assert [row[1] for row in sandbox.steps()] == ["preflight", *STEP_NAMES[:3]]
    stages_run = sandbox.data / "runs" / "r4"
    assert f"STAGES_RUN={stages_run}" in (sandbox.out / "state.env").read_text(encoding="utf-8")
    sandbox.calls.unlink()
    rest = sandbox.run("3")
    assert rest.returncode == 0, rest.stderr
    assert "stopped after step" not in rest.stdout
    assert [row[1] for row in sandbox.steps()] == [
        "preflight", *STEP_NAMES[:3], "preflight", *STEP_NAMES[3:]
    ]  # fmt: skip
    assert _verified(sandbox)[:3] == ["kubric_synthetic", "physics_iq", "pisa_real"]
    assert not _timed(sandbox)
    assert f"query gravity {stages_run}/diagnosis/stages.json" in sandbox.lines()


@pytest.mark.parametrize(
    ("args", "datasets"),
    [
        (("0", "2"), ["kubric_synthetic"]),
        (("1", "1"), ["kubric_synthetic"]),
        (("3", "5"), ["kubric_synthetic", "physics_iq", "pisa_real"]),
        (("0",), ["kubric_synthetic", "physics_iq", "pisa_real"]),
        (("10", "10"), []),
    ],
)
def test_the_preflight_checks_only_the_datasets_of_the_range(
    sandbox: Sandbox, args: tuple[str, ...], datasets: list[str]
) -> None:
    _done(sandbox, int(args[0]))
    result = sandbox.run(*args)
    assert result.returncode == 0, result.stderr
    preflights = [c for c in sandbox.commands() if "--level perceived --subset full" in c]
    assert [c.split()[2] for c in preflights] == datasets
    queried = {line.split()[2] for line in sandbox.lines() if line.startswith("query clips ")}
    assert queried <= {*datasets}


def test_the_pisa_selection_is_checked_only_by_ranges_that_read_it(sandbox: Sandbox) -> None:
    assert sandbox.run("0", "2", PISA_SELECTED="p1").returncode == 0
    result = sandbox.run("3", PISA_SELECTED="p1")
    assert result.returncode == 2
    assert "set PHYSVERIFY_A8V2_PISA_MAX_CLIPS" in result.stderr


def test_a_pisa_extraction_blocks_only_ranges_that_touch_pisa(sandbox: Sandbox) -> None:
    busy = _extracting("pisa_real")
    first = sandbox.run("0", "2", PS_OUT=busy)
    assert first.returncode == 0, first.stderr
    sandbox.calls.unlink()
    later = sandbox.run("3", PS_OUT=busy)
    assert later.returncode == 2
    assert "still running" in later.stderr and "--dataset pisa_real" in later.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize(
    "busy",
    [
        _extracting("local_video"),
        f"4242 {SHARD} extract run --dataset=local_video --subset s",
    ],
)
def test_jobs_outside_the_range_never_block(sandbox: Sandbox, busy: str) -> None:
    result = sandbox.run(PS_OUT=f"  1 init\n{busy}")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "busy",
    [
        f"4242 {SHARD} extract run --subset full",
        f"4242 {SHARD} extract run --dataset",
        f"4242 {SHARD} extract run --dataset $DS --subset full",
        f"4242 {SHARD} extract run --dataset local_video --dataset kubric_synthetic",
        f"4242 {SHARD} dataset build --version v2",
        "4242 physverify dataset migrate pair-collision",
        _extracting("kubric_synthetic"),
    ],
)
def test_an_unreadable_or_kubric_job_blocks_the_kubric_range(sandbox: Sandbox, busy: str) -> None:
    result = sandbox.run("0", "2", PS_OUT=f"  1 init\n{busy}")
    assert result.returncode == 2
    assert "still running" in result.stderr
    assert sandbox.lines() == []


def test_a_second_instance_is_refused(sandbox: Sandbox) -> None:
    sandbox.out.mkdir()
    with (sandbox.out / "run.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = sandbox.run("0", "2")
    assert result.returncode == 2
    assert "another run_a8v2.sh holds" in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("0\tmigrate\t0\t1\t-\n1\textract-synthetic\t0\t1\t-\n", "step 2 has not completed"),
        ("2\tstages\t0\t1\t-\n2\tstages\t2\t1\t-\n", "step 2 has not completed (last exit '2')"),
        ("2\tstages\t1\t1\t-\n", "step 2 has not completed (last exit '1')"),
        (
            "2\tstages\t0\t1\t-\n2\tstages\trunning\t0\t-\n",
            "step 2 has not completed (last exit 'running')",
        ),
    ],
)
def test_a_start_after_an_incomplete_step_is_refused(
    sandbox: Sandbox, rows: str, message: str
) -> None:
    _done(sandbox, 2)
    with (sandbox.out / "steps.tsv").open("a", encoding="utf-8") as tsv:
        tsv.write(rows)
    result = sandbox.run("3")
    assert result.returncode == 2
    assert message in result.stderr and "rerun from" in result.stderr
    assert sandbox.commands() == ["doctor"]


def test_a_step_rerun_successfully_counts_as_completed(sandbox: Sandbox) -> None:
    (sandbox.out).mkdir()
    (sandbox.out / "steps.tsv").write_text("0\tmigrate\t2\t1\t-\n", encoding="utf-8")
    _done(sandbox, 3)
    result = sandbox.run("3", "3")
    assert result.returncode == 0, result.stderr


def test_a_start_without_the_state_of_step_2_is_refused(sandbox: Sandbox) -> None:
    _done(sandbox, 3, saves=0)
    result = sandbox.run("3")
    assert result.returncode == 2
    assert "step 2 recorded no STAGES_RUN" in result.stderr


@pytest.mark.parametrize(
    "busy",
    [
        f"4242 {SHARD} diagnose stages --dataset kubric_synthetic --subset full --write-profile",
        f"4242 {SHARD} --no-lock diagnose stages --dataset kubric_synthetic --clips k1",
        f"4242 {SHARD} diagnose stages --dataset pisa_real",
        f"4242 {SHARD} diagnose real",
        f"4242 {SHARD} diagnose real --dataset all",
    ],
)
def test_a_diagnose_job_on_the_range_blocks(sandbox: Sandbox, busy: str) -> None:
    _done(sandbox, 3)
    result = sandbox.run("3", PS_OUT=busy)
    assert result.returncode == 2
    assert "still running" in result.stderr
    assert sandbox.lines() == []


def test_a_job_started_after_startup_stops_the_next_step(sandbox: Sandbox) -> None:
    result = sandbox.run("0", "2", LATE_PS=_extracting("kubric_synthetic"))
    assert result.returncode == 2
    assert "still running" in result.stderr
    assert sandbox.steps()[-1][:3] == ["1", "extract-synthetic", "0"]
    assert not any("--write-profile" in c for c in sandbox.commands())


def test_a_start_at_7_checks_the_pisa_selection(sandbox: Sandbox) -> None:
    _done(sandbox, 7)
    result = sandbox.run("7", PISA_SELECTED="p1")
    assert result.returncode == 2
    assert "set PHYSVERIFY_A8V2_PISA_MAX_CLIPS" in result.stderr


CUDA = {"PHYSVERIFY_A8V2_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": "3"}
CUDA_PREFIX = re.compile(r"^--profile precise --device cuda --set runtime\.threads=\d+ ")


def _envs(sandbox: Sandbox) -> set[str]:
    return {line.removeprefix("env ") for line in sandbox.lines() if line.startswith("env ")}


def test_the_cpu_chain_hides_the_gpus_and_keeps_the_shared_paths(sandbox: Sandbox) -> None:
    assert sandbox.run("0", "2", CUDA_VISIBLE_DEVICES="0").returncode == 0
    assert _envs(sandbox) == {f"|unset|unset|{sandbox.root / 'kubric.lock'}"}
    assert not (sandbox.root / "a8v2-cuda").exists()


def test_the_cuda_chain_writes_only_its_own_artifacts(sandbox: Sandbox) -> None:
    _audit(sandbox)
    result = sandbox.run(REAL_QUERY="1", **CUDA)
    assert result.returncode == 0, result.stderr
    out = sandbox.root / "a8v2-cuda"
    pv_lines = [line[3:] for line in sandbox.lines() if line.startswith("pv ")]
    assert pv_lines and all(CUDA_PREFIX.match(line) for line in pv_lines)
    calibration, runs = sandbox.data / "calibration-cuda", sandbox.data / "runs-cuda"
    assert _envs(sandbox) == {f"3|{calibration}|{runs}|{out / 'heavy.lock'}"}
    assert not sandbox.out.exists()
    assert [row[1] for row in (line.split("\t") for line in _rows(out))][-1] == "gravity"
    assert not (sandbox.data / "runs" / "r1").exists() and (runs / "r1").is_dir()
    decide = next(line for line in pv_lines if " exclusions decide " in line)
    assert f"--evaluation {runs / 'r10' / 'evaluation.json'} " in decide
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("# A8 v2 (CUDA)") and "already complete" in summary


def _rows(out: Path) -> list[str]:
    return (out / "steps.tsv").read_text(encoding="utf-8").splitlines()


def test_the_cuda_chain_never_runs_the_shared_migration(sandbox: Sandbox) -> None:
    result = sandbox.run("0", "0", **CUDA)
    assert result.returncode == 2
    assert "run step 0 of the cpu chain" in result.stderr
    assert not any(c.startswith("migrate") for c in sandbox.commands())
    assert _rows(sandbox.root / "a8v2-cuda")[-1].split("\t")[:3] == ["0", "migrate", "2"]


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_A8V2_DEVICE": "gpu"}, "must be cpu or cuda"),
        ({**CUDA, "CUDA_VISIBLE_DEVICES": ""}, "must name the one GPU"),
        ({**CUDA, "CUDA_VISIBLE_DEVICES": "1,3"}, "must name the one GPU"),
    ],
)
def test_a_bad_device_setting_stops_before_any_call(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2
    assert message in result.stderr
    assert sandbox.lines() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"DOCTOR_DEVICE": "cpu"}, "not cuda"),
        ({"CAL_DIR": "/elsewhere/calibration"}, "paths.calibration_dir"),
        ({"PROFILE_FILE": "/c/stage_errors.precise.json"}, "outside"),
    ],
)
def test_cuda_paths_shared_with_the_cpu_chain_stop_after_doctor(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run(**CUDA, **env)
    assert result.returncode == 2
    assert message in result.stderr
    assert sandbox.commands() == ["--profile precise --device cuda --set runtime.threads=32 doctor"]


CUDA_SHARD = SHARD.replace("--device cpu", "--device cuda")
KUBRIC_JOB = "extract run --dataset kubric_synthetic --subset full"


@pytest.mark.parametrize(
    ("device", "busy"),
    [
        ("cpu", f"4242 {CUDA_SHARD} {KUBRIC_JOB}"),
        ("cpu", f"4242 {SHARD.replace('--device cpu', '--device=cuda:1')} {KUBRIC_JOB}"),
        ("cpu", f"4242 {SHARD} --set runtime.device=cuda {KUBRIC_JOB}"),
        ("cuda", f"4242 {SHARD} {KUBRIC_JOB}"),
        ("cuda", f"4242 {SHARD} --no-lock diagnose stages --dataset kubric_synthetic --clips k1"),
    ],
)
def test_jobs_of_the_other_device_never_block(sandbox: Sandbox, device: str, busy: str) -> None:
    env = CUDA if device == "cuda" else {}
    _audit(sandbox)
    if device == "cuda":
        sandbox.out = sandbox.root / "a8v2-cuda"
    _done(sandbox, 1)
    result = sandbox.run("1", "2", PS_OUT=f"  1 init\n{busy}", **env)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "busy",
    [
        f"4242 {CUDA_SHARD} {KUBRIC_JOB}",
        f"4242 {SHARD.replace('--device cpu', '--device cuda:0')} {KUBRIC_JOB}",
        f"4242 {SHARD} --set runtime.device=cuda {KUBRIC_JOB}",
        f"4242 python -c from physverify.cli.main import main; main() {KUBRIC_JOB}",
        f"4242 {SHARD} --device {KUBRIC_JOB}",
        f"4242 {SHARD} dataset build --version v2",
    ],
)
def test_same_device_unreadable_and_dataset_jobs_block_the_cuda_chain(
    sandbox: Sandbox, busy: str
) -> None:
    result = sandbox.run("0", "2", PS_OUT=f"  1 init\n{busy}", **CUDA)
    assert result.returncode == 2
    assert "still running" in result.stderr
    assert sandbox.lines() == []


def test_a_run_from_step_0_blocks_on_jobs_of_either_device(sandbox: Sandbox) -> None:
    busy = f"4242 {SHARD} --device cuda extract run --dataset kubric_synthetic --clips k1"
    result = sandbox.run("0", "2", PS_OUT=busy)
    assert result.returncode == 2
    assert "still running" in result.stderr


def test_an_output_dir_of_the_other_chain_is_refused(sandbox: Sandbox) -> None:
    sandbox.out.mkdir()
    (sandbox.out / "state.env").write_text("CHAIN_DEVICE=cuda\n", encoding="utf-8")
    result = sandbox.run("0", "2")
    assert result.returncode == 2
    assert "belongs to the cuda chain" in result.stderr
