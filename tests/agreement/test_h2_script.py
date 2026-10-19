"""scripts/h200/run_h2.sh against a fake ``uv`` and ``nvidia-smi`` in a copied tree (C9)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from check_rows import UNCACHED, write_check_rows

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "setup.sh", "run_h2.sh", "a8v2_query.py")
TOOLS = (
    "bash", "basename", "cat", "cmp", "cp", "dirname", "grep", "head", "ln", "mkdir", "mktemp",
    "mv", "nproc", "rm", "sed", "tee",
)  # fmt: skip
PV = "uv run --no-sync physverify --profile precise --device cuda --set runtime.threads=2 "
PV_CPU = PV.replace("--device cuda", "--device cpu")
PREFLIGHT = (
    f"{PV_CPU}verify --dataset videophy --level perceived --subset h2_dev --clips v1-a8208f4dd41e"
)

# Each extract/verify call creates the next numbered run and prints it like the real commands; the
# preflight verify writes its checks.jsonl, which the real helper reads unless they are the clean
# rows and REAL_QUERY is unset (its import costs seconds).
FAKE_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$CALLS"
[[ "${PHYSVERIFY__CHECK_SCOPE__REQUIRE_DECISION_RECORD:-}" == true ]] \\
    || echo "no record" >> "$CALLS"
case "$*" in
  "run --no-sync python "*)
    [[ -z "${REAL_QUERY:-}" ]] && cmp -s "$CLEAN_CHECKS" "${!#}/checks.jsonl" && exit 0
    shift 3; PYTHONPATH="$REAL_SRC" exec "$REAL_PYTHON" "$@" ;;
  *" --clips "*)
    n=$(( $(cat "$COUNTER" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$COUNTER"
    mkdir -p "$RUNS/run$n"; echo "run: $RUNS/run$n"
    cp "${PREFLIGHT_CHECKS:-$CLEAN_CHECKS}" "$RUNS/run$n/checks.jsonl"
    exit "${PREFLIGHT_EXIT:-0}" ;;
  *" extract run "*|*" verify "*)
    n=$(( $(cat "$COUNTER" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$COUNTER"
    mkdir -p "$RUNS/run$n"; echo "run: $RUNS/run$n"; exit "${CLIP_EXIT:-0}" ;;
  *" agreement subset --check"*) exit "${CHECK_EXIT:-0}" ;;
  *" agreement freeze "*)
    state="${FROZEN_STATE:-state-a}"
    if [[ -f "$RUNS/prereg.json" && "$(cat "$RUNS/prereg.json")" != "$state" ]]; then
      echo "prereg.json exists with a different frozen state" >&2; exit 2
    fi
    [[ -f "$RUNS/prereg.json" ]] || echo "$state" > "$RUNS/prereg.json"
    echo "preregistration: $RUNS/prereg.json" ;;
  *" agreement compare "*) exit "${COMPARE_EXIT:-0}" ;;
esac
"""


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.repo = root / "repo"
        self.runs = self.repo / "data" / "runs"
        self.calls = root / "calls.log"
        scripts = self.repo / "scripts" / "h200"
        scripts.mkdir(parents=True)
        for name in SCRIPT_NAMES:
            shutil.copy2(SCRIPTS / name, scripts / name)
        tools, self.fakes = root / "tools", root / "fakes"
        tools.mkdir()
        self.fakes.mkdir()
        for tool in TOOLS:
            found = shutil.which(tool)
            assert found is not None, tool
            (tools / tool).symlink_to(found)
        self.bash = tools / "bash"
        self.uncached = write_check_rows(root / "uncached.jsonl", error=UNCACHED)
        self.env = {
            "REAL_PYTHON": sys.executable,
            "REAL_SRC": str(REPO / "src"),
            "CLEAN_CHECKS": str(write_check_rows(root / "clean.jsonl")),
            "PATH": f"{self.fakes}:{tools}",
            "HOME": str(root),
            "TMPDIR": str(root),
            "CALLS": str(self.calls),
            "COUNTER": str(root / "counter"),
            "RUNS": str(self.runs),
            "PHYSVERIFY_TIMING_THREADS": "2",
        }

    def install(self, *, nvidia_smi: bool = True) -> None:
        _executable(self.fakes / "uv", FAKE_UV)
        if nvidia_smi:
            _executable(self.fakes / "nvidia-smi", "#!/bin/bash\necho 'GPU 0: NVIDIA H200'\n")

    def run(self, **env: str) -> subprocess.CompletedProcess[str]:
        script = self.repo / "scripts" / "h200" / "run_h2.sh"
        return subprocess.run(
            [str(self.bash), str(script)],
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def uv_calls(self) -> list[str]:
        if not self.calls.exists():
            return []
        return self.calls.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def _pass(subset: str) -> list[str]:
    return [
        f"{PV_CPU}agreement fetch --subset {subset}",
        f"{PV_CPU}extract run --dataset videophy --subset {subset}",
        f"{PV_CPU}verify --dataset videophy --level perceived --subset {subset}",
    ]


def test_the_script_parses_and_is_executable() -> None:
    subprocess.run(["bash", "-n", str(SCRIPTS / "run_h2.sh")], check=True)
    assert os.access(SCRIPTS / "run_h2.sh", os.X_OK)


def test_without_a_gpu_nothing_touches_uv(sandbox: Sandbox) -> None:
    sandbox.install(nvidia_smi=False)
    result = sandbox.run()
    assert result.returncode == 2 and "H200" in result.stderr
    assert sandbox.uv_calls() == []


def test_dev_freeze_then_test_compare_and_the_latest_link(sandbox: Sandbox) -> None:
    sandbox.install()
    result = sandbox.run(CLIP_EXIT="1", REAL_QUERY="1")
    assert result.returncode == 0, result.stderr
    calls = [c for c in sandbox.uv_calls() if "physverify" in c and "fetch-weights" not in c]
    assert calls[-10:] == [
        f"{PV}agreement subset --check",
        PREFLIGHT,
        *_pass("h2_dev"),
        f"{PV}agreement freeze --verify-run {sandbox.runs}/run3",
        *_pass("h2_test"),
        f"{PV}agreement compare --verify-run {sandbox.runs}/run5 --subset h2_test",
    ]
    link = sandbox.runs / "h2-latest"
    assert link.is_symlink() and os.readlink(link) == "run5"
    assert (sandbox.runs / "run1.log").is_file()
    assert (sandbox.runs / "run3" / "freeze.log").is_file()
    assert (sandbox.runs / "run5" / "agreement.log").is_file()
    assert (sandbox.runs / "run5" / "agreement" / "preregistration.json").is_file()
    assert (sandbox.runs / "run4.log").read_text(encoding="utf-8").startswith("run: ")


def test_every_step_requires_the_exclusion_record(sandbox: Sandbox) -> None:
    sandbox.install()
    assert sandbox.run().returncode == 0
    assert sandbox.calls.read_text(encoding="utf-8").count("no record") == 0


def test_an_uncached_preflight_clip_stops_before_any_extraction(sandbox: Sandbox) -> None:
    """verify exits 0 on an uncached clip; only its checks.jsonl carries the error."""
    sandbox.install()
    result = sandbox.run(PREFLIGHT_CHECKS=str(sandbox.uncached))
    assert result.returncode == 2 and "preflight" in result.stderr
    assert "heavy stage is not cached" in result.stderr
    assert not any(" extract run " in call for call in sandbox.uv_calls())


def test_a_failed_preflight_clip_with_its_cache_is_not_fatal(sandbox: Sandbox) -> None:
    sandbox.install()
    result = sandbox.run(PREFLIGHT_EXIT="1")
    assert result.returncode == 0, result.stderr
    assert any("a8v2_query.py run-errors" in call for call in sandbox.uv_calls())


def test_the_h2_device_can_be_overridden(sandbox: Sandbox) -> None:
    sandbox.install()
    assert sandbox.run(PHYSVERIFY_H2_DEVICE="cuda").returncode == 0
    assert f"{PV}extract run --dataset videophy --subset h2_dev" in sandbox.uv_calls()


def test_a_failed_compare_stops_before_the_link(sandbox: Sandbox) -> None:
    sandbox.install()
    result = sandbox.run(COMPARE_EXIT="2")
    assert result.returncode == 2
    assert not (sandbox.runs / "h2-latest").exists()


def test_an_unrecorded_review_stops_before_any_download(sandbox: Sandbox) -> None:
    sandbox.install()
    result = sandbox.run(CHECK_EXIT="1")
    assert result.returncode == 2 and "caption review" in result.stderr
    assert not any(" fetch " in call for call in sandbox.uv_calls())


def test_rerunning_with_the_same_frozen_state_reuses_the_preregistration(
    sandbox: Sandbox,
) -> None:
    sandbox.install()
    assert sandbox.run().returncode == 0
    again = sandbox.run()
    assert again.returncode == 0, again.stderr
    assert os.readlink(sandbox.runs / "h2-latest") == "run10"
    copied = sandbox.runs / "run10" / "agreement" / "preregistration.json"
    assert copied.read_text(encoding="utf-8").strip() == "state-a"


def test_rerunning_with_a_different_frozen_state_stops_before_the_test_pass(
    sandbox: Sandbox,
) -> None:
    sandbox.install()
    assert sandbox.run().returncode == 0
    changed = sandbox.run(FROZEN_STATE="state-b")
    assert changed.returncode == 2
    assert sandbox.uv_calls()[-1].endswith(f"agreement freeze --verify-run {sandbox.runs}/run8")
    assert os.readlink(sandbox.runs / "h2-latest") == "run5"
