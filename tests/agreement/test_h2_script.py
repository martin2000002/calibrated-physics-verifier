"""scripts/h200/run_h2.sh against a fake ``uv`` and ``nvidia-smi`` in a copied tree (C9)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "h200"
SCRIPT_NAMES = ("_common.sh", "setup.sh", "run_h2.sh")
TOOLS = (
    "bash", "basename", "cat", "cp", "dirname", "grep", "head", "ln", "mkdir", "mktemp", "mv",
    "nproc", "rm", "sed", "tee",
)  # fmt: skip
PV = "uv run --no-sync physverify --profile precise --device cuda --set runtime.threads=2 "

# Each extract/verify call creates the next numbered run and prints it like the real commands.
FAKE_UV = """#!/bin/bash
printf 'uv %s\\n' "$*" >> "$CALLS"
case "$*" in
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
        self.env = {
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
        f"{PV}agreement fetch --subset {subset}",
        f"{PV}extract run --dataset videophy --subset {subset}",
        f"{PV}verify --dataset videophy --level perceived --subset {subset}",
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
    result = sandbox.run(CLIP_EXIT="1")
    assert result.returncode == 0, result.stderr
    calls = [c for c in sandbox.uv_calls() if "physverify" in c and "fetch-weights" not in c]
    assert calls[-9:] == [
        f"{PV}agreement subset --check",
        *_pass("h2_dev"),
        f"{PV}agreement freeze --verify-run {sandbox.runs}/run2",
        *_pass("h2_test"),
        f"{PV}agreement compare --verify-run {sandbox.runs}/run4 --subset h2_test",
    ]
    link = sandbox.runs / "h2-latest"
    assert link.is_symlink() and os.readlink(link) == "run4"
    assert (sandbox.runs / "run2" / "freeze.log").is_file()
    assert (sandbox.runs / "run4" / "agreement.log").is_file()
    assert (sandbox.runs / "run4" / "agreement" / "preregistration.json").is_file()
    assert (sandbox.runs / "run3.log").read_text(encoding="utf-8").startswith("run: ")


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
    assert os.readlink(sandbox.runs / "h2-latest") == "run8"
    copied = sandbox.runs / "run8" / "agreement" / "preregistration.json"
    assert copied.read_text(encoding="utf-8").strip() == "state-a"


def test_rerunning_with_a_different_frozen_state_stops_before_the_test_pass(
    sandbox: Sandbox,
) -> None:
    sandbox.install()
    assert sandbox.run().returncode == 0
    changed = sandbox.run(FROZEN_STATE="state-b")
    assert changed.returncode == 2
    assert sandbox.uv_calls()[-1].endswith(f"agreement freeze --verify-run {sandbox.runs}/run6")
    assert os.readlink(sandbox.runs / "h2-latest") == "run4"
