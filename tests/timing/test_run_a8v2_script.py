"""scripts/h200/run_a8v2.sh against a fake interpreter, ``ps``, ``git`` and migration script."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

H200 = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT = H200 / "run_a8v2.sh"
TOOLS = (
    "basename", "bash", "cat", "dirname", "env", "grep", "head", "mkdir", "mv", "nproc",
    "realpath", "sed", "sort", "tee", "touch", "tr",
)  # fmt: skip
PREFIX = re.compile(r"^--profile precise --device cpu --set runtime\.threads=\d+ ")
FAKE_PYTHON = r"""#!/bin/bash
if [[ "$1" != -c ]]; then
    shift
    printf 'query %s\n' "$*" >> "$CALLS"
    case "$*" in
      *"clips pisa_real full") printf '%s\n' ${PISA_SELECTED:-p1 p2} ;;
      *"clips kubric_synthetic full --split calib") printf 'k1\nk2\nk3\n' ;;
      *"clips physics_iq full") echo q1 ;;
      *"rerendered "*) echo k1 ;;
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
    echo "hash$n" > "$RUN/checks_hash.txt"
    echo "run: $RUN"
}
case "$*" in
  *" doctor")
    echo "paths.cache_dir: ${CACHE:-$DATA/cache} (env)"
    echo "paths.synthetic_dir: $DATA/synthetic"
    echo "heavy_lock: $PHYSVERIFY__RUNTIME__HEAVY_LOCK (free)" ;;
  *" verify "*)
    new_run
    if [[ -n "${NOT_CACHED:-}" ]]; then echo "a heavy stage is not cached"; exit 1; fi ;;
  *" extract run "*) new_run; exit "${EXTRACT_EXIT:-0}" ;;
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
FAKE_GIT = "#!/bin/bash\necho c0ffee1234567\n"
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


def test_a_cache_other_than_the_shards_stops_after_doctor(sandbox: Sandbox) -> None:
    result = sandbox.run(CACHE="/elsewhere/cache")
    assert result.returncode == 2
    assert "not the shards'" in result.stderr
    assert sandbox.commands() == ["doctor"]


@pytest.mark.parametrize("start", ["x", "12", "-1", "01"])
def test_a_bad_start_step_is_a_usage_error(sandbox: Sandbox, start: str) -> None:
    result = sandbox.run(start)
    assert result.returncode == 2
    assert sandbox.lines() == []


def test_the_full_run_preflights_then_runs_every_step_in_order(sandbox: Sandbox) -> None:
    result = sandbox.run()
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
    result = sandbox.run(NOT_CACHED="1")
    assert result.returncode == 2
    assert "is not cached" in result.stderr
    assert not any(c.startswith("migrate") for c in sandbox.commands())


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
