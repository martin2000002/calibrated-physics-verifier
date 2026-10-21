"""scripts/h200/run_iteration.sh against a fake interpreter and ``git`` (WP-O AC17-AC20)."""

from __future__ import annotations

import fcntl
import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
H200 = REPO / "scripts" / "h200"
SCRIPT_NAMES = (
    "_common.sh",
    "_steps.sh",
    "run_iteration.sh",
    "a8v2_query.py",
    "iteration_arms.tsv",
)
TOOLS = (
    "awk", "basename", "bash", "cat", "cp", "cut", "dirname", "env", "flock", "grep", "mkdir",
    "mv", "nproc", "realpath", "sed", "sha256sum", "tee", "touch",
)  # fmt: skip
PREFIX = re.compile(
    r"^--profile [a-z_]+ --device (cpu|cuda) --set runtime\.threads=16 "
    r"(--set extraction\.plan\.[a-z]+=[a-z0-9_]+ )*"
)
CALIB = ("k-calib-1", "k-calib-2")
CLIPS = " ".join(f"--clips {c}" for c in CALIB)
KS = "--dataset kubric_synthetic --subset full"
# pv calls are logged with the paths they see; doctor answers from the environment unless a
# DOCTOR_* variable overrides a line; stages, stage-sigma and devscore write what they would.
FAKE_PYTHON = r"""#!/bin/bash
CAL="$PHYSVERIFY__PATHS__CALIBRATION_DIR"
if [[ "$1" != -c ]]; then
    printf 'query %s\n' "${*:2}" >> "$CALLS"
    [[ "$*" == *"clips kubric_synthetic full --split calib" ]] && printf 'k-calib-1\nk-calib-2\n'
    exit 0
fi
shift 2
printf 'pv %s\n' "$*" >> "$CALLS"
printf 'env %s|%s|%s|%s|%s\n' "${CUDA_VISIBLE_DEVICES-unset}" "$PHYSVERIFY__PATHS__CACHE_DIR" \
    "$CAL" "$PHYSVERIFY__PATHS__RUNS_DIR" "$PHYSVERIFY__RUNTIME__HEAVY_LOCK" >> "$CALLS"
new_run() {
    local n
    n=$(( $(cat "$PHYSVERIFY__PATHS__RUNS_DIR/counter" 2>/dev/null || echo 0) + 1 ))
    mkdir -p "$PHYSVERIFY__PATHS__RUNS_DIR/r$n"
    echo "$n" > "$PHYSVERIFY__PATHS__RUNS_DIR/counter"
    RUN="$PHYSVERIFY__PATHS__RUNS_DIR/r$n"
    echo "run: $RUN"
}
case "$*" in
  *" doctor")
    for arg in "$@"; do
        [[ "$arg" == extraction.plan.* && -z "${DOCTOR_DROP_SETS:-}" ]] && echo "source: --set $arg"
    done
    echo "profile: ${DOCTOR_PROFILE:-$2}"
    echo "device: ${DOCTOR_DEVICE:-${4}}"
    echo "config_dir: ${DOCTOR_CONFIG_DIR:-$REPO_CONFIGS}"
    echo "paths.cache_dir: ${DOCTOR_CACHE_DIR:-$PHYSVERIFY__PATHS__CACHE_DIR} (env)"
    echo "paths.runs_dir: ${DOCTOR_RUNS_DIR:-$PHYSVERIFY__PATHS__RUNS_DIR}"
    echo "paths.calibration_dir: ${DOCTOR_CALIBRATION_DIR:-$CAL}"
    echo "paths.error_profile: ${DOCTOR_ERROR_PROFILE:-$CAL/stage_errors.precise.json}"
    echo "paths.thresholds: ${DOCTOR_THRESHOLDS:-$CAL/thresholds.json}" ;;
  *" extract run "*) exit "${CLIP_EXIT:-0}" ;;
  *" diagnose stages "*)
    new_run; mkdir -p "$CAL"; echo profile > "$CAL/stage_errors.precise.json" ;;
  *" calibrate stage-sigma "*) mkdir -p "$CAL"; echo sigma > "$CAL/stage_sigma.json" ;;
  *" calibrate devscore "*)
    [[ -z "${DEVSCORE_EXIT:-}" ]] || exit "$DEVSCORE_EXIT"
    new_run
    echo '{"strata": []}' > "$RUN/calibration.json"
    mkdir -p "$(dirname "${!#}")"; echo '{"levels": []}' > "${!#}"
    echo "perceived/precise: 1/9 meet H1 and J_cv^lo > 0 (R4_jump); AUROC_cv 0.6 [0.5, 0.7]"
    echo "selection (perceived): n_met 1" ;;
esac
exit 0
"""
FAKE_GIT = """#!/bin/bash
case "$*" in
  *" rev-parse HEAD") echo "${HEAD_SHA:-c0ffee1234567}" ;;
  *" status --porcelain "*) printf '%s' "${DIRTY_FILES:-}" ;;
esac
"""
FORBIDDEN = '#!/bin/bash\necho "forbidden $(basename "$0") $*" >> "$CALLS"\n'
EXPECTED = [
    "doctor",
    f"extract run {KS} {CLIPS}",
    f"diagnose stages {KS} {CLIPS} --write-profile",
    f"extract run {KS} {CLIPS}",
    f"calibrate stage-sigma {KS}",
]


def _executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.calls = self.root / "calls.log"
        self.data = self.root / "kubric-data"
        self.out = self.root / "iterations" / "it0-baseline-cpu"
        scripts = self.root / "repo" / "scripts" / "h200"
        scripts.mkdir(parents=True)
        (self.root / "repo" / "configs").mkdir()
        self.script = scripts / "run_iteration.sh"
        for name in SCRIPT_NAMES:
            shutil.copy2(H200 / name, scripts / name)
        tools, fakes = self.root / "tools", self.root / "fakes"
        tools.mkdir()
        fakes.mkdir()
        for tool in TOOLS:
            found = shutil.which(tool)
            assert found is not None, tool
            (tools / tool).symlink_to(found)
        _executable(fakes / "python", FAKE_PYTHON)
        _executable(fakes / "git", FAKE_GIT)
        for name in ("uv", "nvidia-smi"):
            _executable(fakes / name, FORBIDDEN)
        self.env = {
            "PATH": f"{fakes}:{tools}",
            "HOME": str(self.root),
            "CALLS": str(self.calls),
            "REPO_CONFIGS": str(self.root / "repo" / "configs"),
            "PHYSVERIFY_H200_ROOT": str(self.root),
            "PHYSVERIFY_H200_ENV": str(self.root / "no-env.sh"),
            "PHYSVERIFY_ITER_ID": "it0-baseline",
            "PHYSVERIFY_ITER_EXPECT_COMMIT": "c0ffee1",
            "PHYSVERIFY_ITER_CACHE_DIR": str(self.root / "cache"),
            "PHYSVERIFY_ITER_PYTHON": str(fakes / "python"),
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

    def pv(self) -> list[str]:
        return [line[3:] for line in self.lines() if line.startswith("pv ")]

    def commands(self) -> list[str]:
        return [PREFIX.sub("", line) for line in self.pv()]

    def state(self) -> dict[str, str]:
        text = (self.out / "state.env").read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in text.splitlines())


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


def _devscore(out: Path) -> str:
    return f"calibrate devscore {KS} --level both --out {out / 'devscore.json'}"


def test_the_script_parses_and_is_executable() -> None:
    """AC17."""
    for name in ("run_iteration.sh", "_steps.sh"):
        subprocess.run(["bash", "-n", str(H200 / name)], check=True)
    assert os.access(H200 / "run_iteration.sh", os.X_OK)


def test_the_iteration_runs_its_five_steps_on_calib_only(sandbox: Sandbox) -> None:
    """AC19: the exact sequence, one prefix, calib ids only, every path inside $OUT."""
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == [*EXPECTED, _devscore(sandbox.out)]
    assert all(PREFIX.match(line) for line in sandbox.pv())
    every = " ".join(sandbox.lines())
    assert not re.search(r"evaluate|abstain|--real|physics_iq|pisa_real|forbidden", every)
    assert re.findall(r"--clips (\S+)", every) == [*CALIB] * 3
    out = str(sandbox.out)
    for line in (x for x in sandbox.lines() if x.startswith("env ")):
        cuda, cache, *mine = line[4:].split("|")
        assert cuda == "" and cache == str(sandbox.root / "cache")
        assert all(path.startswith(out + "/") for path in mine)
    state = sandbox.state()
    assert state["CHAIN_DEVICE"] == "cpu" and state["CHAIN_ITER_ID"] == "it0-baseline"
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert state["DEVSCORE_SHA"] in summary and state["CALIBRATION_SHA"] in summary
    assert state["PROFILE_SHA"] in summary and "1/9 meet H1" in summary
    assert (sandbox.out / "calibration.json").is_file()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_ITER_ID": ""}, "PHYSVERIFY_ITER_ID"),
        ({"PHYSVERIFY_ITER_ID": "baseline"}, "PHYSVERIFY_ITER_ID"),
        ({"PHYSVERIFY_ITER_EXPECT_COMMIT": ""}, "PHYSVERIFY_ITER_EXPECT_COMMIT"),
        ({"HEAD_SHA": "deadbeef0000"}, "expected c0ffee1"),
        ({"DIRTY_FILES": " M configs/sealed.toml\n"}, "uncommitted or untracked"),
        ({"PHYSVERIFY__SEALED__SETS": "[]"}, "PHYSVERIFY__SEALED__SETS"),
        ({"PHYSVERIFY__PATHS__THRESHOLDS": "/t.json"}, "PHYSVERIFY__PATHS__THRESHOLDS"),
        ({"PHYSVERIFY_UNSEAL": "kubric_synthetic:eval"}, "PHYSVERIFY_UNSEAL is set"),
        ({"PHYSVERIFY_UNSEAL_REPEAT": "1"}, "PHYSVERIFY_UNSEAL_REPEAT is set"),
        ({"PHYSVERIFY_CONFIG_DIR": "/other"}, "PHYSVERIFY_CONFIG_DIR"),
        ({"PHYSVERIFY_ITER_CACHE_DIR": ""}, "PHYSVERIFY_ITER_CACHE_DIR"),
        ({"PHYSVERIFY_ITER_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": "5"}, "CUDA_VISIBLE_DEVICES"),
        ({"PHYSVERIFY_ITER_DEVICE": "tpu"}, "cpu or cuda"),
        ({"PHYSVERIFY_ITER_THREADS": "0"}, "PHYSVERIFY_ITER_THREADS"),
    ],
)
def test_a_startup_guard_stops_before_any_call(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    """AC17."""
    result = sandbox.run(**env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.pv() == []


@pytest.mark.parametrize(
    "target",
    ["a8v2", "a8v2-cuda/it", "kubric-data/calibration", "kubric-data/runs/x", "kubric-data"],
)
def test_an_output_near_the_chain_or_its_artifacts_is_refused(
    sandbox: Sandbox, target: str
) -> None:
    """AC17: $OUT, $OUT/calibration and $OUT/runs stay away from A8 v2's files."""
    result = sandbox.run(PHYSVERIFY_ITER_DIR=str(sandbox.root / target))
    assert result.returncode == 2 and "never writes there" in result.stderr, result.stderr
    assert sandbox.pv() == []


def test_local_toml_and_a_held_lock_stop_before_any_call(sandbox: Sandbox) -> None:
    """AC17."""
    local = sandbox.root / "repo" / "configs" / "local.toml"
    local.write_text("", encoding="utf-8")
    assert sandbox.run().returncode == 2
    local.unlink()
    sandbox.out.mkdir(parents=True)
    with (sandbox.out / "run.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = sandbox.run()
    assert result.returncode == 2 and "run.lock" in result.stderr
    assert sandbox.pv() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"DOCTOR_CALIBRATION_DIR": "/elsewhere/calibration"}, "paths.calibration_dir"),
        ({"DOCTOR_RUNS_DIR": "/elsewhere/runs"}, "paths.runs_dir"),
        ({"DOCTOR_ERROR_PROFILE": "/e/stage_errors.precise.json"}, "paths.error_profile"),
        ({"DOCTOR_THRESHOLDS": "/e/thresholds.json"}, "paths.thresholds"),
        ({"DOCTOR_DEVICE": "cuda"}, "device"),
        ({"DOCTOR_PROFILE": "fast"}, "profile"),
        ({"DOCTOR_CACHE_DIR": "/other/cache"}, "paths.cache_dir"),
        ({"DOCTOR_CONFIG_DIR": "/other/configs"}, "config_dir"),
    ],
)
def test_doctor_must_confirm_the_iteration_paths(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    """AC18: exit 2 after doctor only."""
    result = sandbox.run(**env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == ["doctor"]


def test_resume_runs_only_the_range_and_needs_the_earlier_steps(sandbox: Sandbox) -> None:
    """AC20."""
    fresh = sandbox.run("3")
    assert fresh.returncode == 2 and "step 1 has not completed" in fresh.stderr
    assert sandbox.pv() == []
    assert sandbox.run("1", "2").returncode == 0
    sandbox.calls.unlink()
    result = sandbox.run("3")
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == ["doctor", EXPECTED[3], EXPECTED[4], _devscore(sandbox.out)]


@pytest.mark.parametrize(
    ("key", "value"),
    [("CHAIN_DEVICE", "cuda"), ("CHAIN_PROFILE", "fast"), ("CHAIN_ITER_ID", "it9")],
)
def test_resume_with_another_device_profile_or_id_is_refused(
    sandbox: Sandbox, key: str, value: str
) -> None:
    """AC20."""
    sandbox.out.mkdir(parents=True)
    (sandbox.out / "state.env").write_text(f"{key}={value}\n", encoding="utf-8")
    result = sandbox.run()
    assert result.returncode == 2 and f"belongs to {key}={value}" in result.stderr
    assert sandbox.pv() == []


def test_a_cuda_iteration_has_its_own_directory_and_device(sandbox: Sandbox) -> None:
    result = sandbox.run(PHYSVERIFY_ITER_DEVICE="cuda", CUDA_VISIBLE_DEVICES="2")
    assert result.returncode == 0, result.stderr
    out = sandbox.root / "iterations" / "it0-baseline-cuda"
    assert all("--device cuda" in line for line in sandbox.pv())
    assert all(line.startswith("env 2|") for line in sandbox.lines() if line.startswith("env "))
    assert (out / "devscore.json").is_file()


def test_failed_clips_are_noted_and_a_failed_devscore_stops(sandbox: Sandbox) -> None:
    assert sandbox.run(CLIP_EXIT="1").returncode == 0
    notes = (sandbox.out / "notes.md").read_text(encoding="utf-8")
    assert "some clips failed (exit 1)" in notes
    other = Sandbox(sandbox.root / "second")
    result = other.run(DEVSCORE_EXIT="1")
    assert result.returncode == 1 and "DEVSCORE_SHA" not in other.state()


def test_the_baseline_is_protected_whatever_data_is(sandbox: Sandbox) -> None:
    """Audit m5: with another PHYSVERIFY_ITER_DATA, kubric-data/calibration is still refused."""
    result = sandbox.run(
        PHYSVERIFY_ITER_DATA=str(sandbox.root / "other-data"),
        PHYSVERIFY_ITER_DIR=str(sandbox.root / "kubric-data"),
    )
    assert result.returncode == 2 and "never writes there" in result.stderr, result.stderr
    assert sandbox.pv() == []


def test_another_profile_reaches_every_call_and_the_clip_list(sandbox: Sandbox) -> None:
    """PHYSVERIFY_ITER_PROFILE: every pv call, the calib query and state.env use it."""
    result = sandbox.run(PHYSVERIFY_ITER_PROFILE="gpu_strong")
    assert result.returncode == 0, result.stderr
    assert all(line.startswith("--profile gpu_strong ") for line in sandbox.pv())
    queries = [line for line in sandbox.lines() if line.startswith("query ")]
    assert queries == ["query --profile gpu_strong clips kubric_synthetic full --split calib"]
    assert sandbox.state()["CHAIN_PROFILE"] == "gpu_strong"
    resumed = sandbox.run("5")
    assert resumed.returncode == 2 and "belongs to CHAIN_PROFILE=gpu_strong" in resumed.stderr


def test_a_bad_profile_name_and_an_exported_pv_profile(sandbox: Sandbox) -> None:
    """Audit m4: an exported PV_PROFILE never reaches pv; the profile is explicit."""
    bad = sandbox.run(PHYSVERIFY_ITER_PROFILE="Gpu;x")
    assert bad.returncode == 2 and "PHYSVERIFY_ITER_PROFILE" in bad.stderr
    assert sandbox.run(PV_PROFILE="fast").returncode == 0
    assert all(line.startswith("--profile precise ") for line in sandbox.pv())


def test_steps_sh_ignores_an_exported_pv_profile() -> None:
    """Audit m4: run_h2.sh sources _steps.sh and must keep --profile precise."""
    script = f'set -u; die() {{ exit 2; }}; source "{H200 / "_steps.sh"}"; echo "$PV_PROFILE"'
    found = subprocess.run(
        ["bash", "-c", script], env={"PV_PROFILE": "fast", "PATH": os.environ["PATH"]},
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert found.stdout.strip() == "precise"


def test_the_query_helper_takes_the_profile() -> None:
    spec = importlib.util.spec_from_file_location("a8v2_query_profile", H200 / "a8v2_query.py")
    assert spec is not None and spec.loader is not None
    query = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(query)
    parser = query._parser()
    assert parser.parse_args(["clips", "ds", "full"]).profile == "precise"
    assert parser.parse_args(["--profile", "fast", "clips", "ds", "full"]).profile == "fast"
    assert query._settings([], "fast").profile == "fast"


ARM_C = ["--set extraction.plan.depth=vda_large", "--set extraction.plan.contacts=kinematic"]


def test_an_arm_swaps_providers_on_every_call_on_one_gpu(sandbox: Sandbox) -> None:
    """it3: PHYSVERIFY_ITER_ARM passes its declared swaps to every physverify call, the run is
    CUDA on GPU 0 only, and state.env and the summary name the arm."""
    result = sandbox.run(
        PHYSVERIFY_ITER_ID="it3-c",
        PHYSVERIFY_ITER_ARM="it3-c",
        PHYSVERIFY_ITER_DEVICE="cuda",
        CUDA_VISIBLE_DEVICES="0",
    )
    assert result.returncode == 0, result.stderr
    out = sandbox.root / "iterations" / "it3-c-cuda"
    swaps = " ".join(ARM_C)
    assert all(f"--device cuda --set runtime.threads=16 {swaps} " in line for line in sandbox.pv())
    assert all(line.startswith("env 0|") for line in sandbox.lines() if line.startswith("env "))
    state = dict(
        line.split("=", 1) for line in (out / "state.env").read_text(encoding="utf-8").splitlines()
    )
    assert state["CHAIN_ARM"] == "it3-c"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert (
        "arm it3-c (extraction.plan.depth=vda_large extraction.plan.contacts=kinematic)" in summary
    )
    assert "CUDA_VISIBLE_DEVICES=0" in summary


def test_the_four_it3_arms_are_declared() -> None:
    table = (H200 / "iteration_arms.tsv").read_text(encoding="utf-8").splitlines()
    arms = dict(line.split("\t") for line in table if line and not line.startswith("#"))
    assert arms == {
        "it3-r": "extraction.plan.depth=vda_small extraction.plan.contacts=proximity",
        "it3-ak": "extraction.plan.depth=vda_small extraction.plan.contacts=kinematic",
        "it3-ap": "extraction.plan.depth=vda_large extraction.plan.contacts=proximity",
        "it3-c": "extraction.plan.depth=vda_large extraction.plan.contacts=kinematic",
    }


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_ITER_ARM": "it3-x"}, "not declared exactly once"),
        ({"PHYSVERIFY_ITER_ARM": "It3;c"}, "must be an arm name"),
        ({"PHYSVERIFY_ITER_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": "4"}, "0-3"),
        ({"PHYSVERIFY_ITER_DEVICE": "cuda", "CUDA_VISIBLE_DEVICES": "0,1"}, "0-3"),
        ({"PHYSVERIFY_ITER_DEVICE": "cuda"}, "unset"),
    ],
)
def test_an_undeclared_arm_or_a_gpu_outside_0_to_3_stops_before_any_call(
    sandbox: Sandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.pv() == []


def test_a_malformed_swap_and_a_doctor_without_the_swaps_stop(sandbox: Sandbox) -> None:
    table = sandbox.script.parent / "iteration_arms.tsv"
    table.write_text("it3-bad\tchecks.R1.tau=0\n", encoding="utf-8")
    bad = sandbox.run(PHYSVERIFY_ITER_ARM="it3-bad")
    assert bad.returncode == 2 and "extraction.plan.<stage>=<provider>" in bad.stderr
    assert sandbox.pv() == []
    table.write_text("it3-ok\textraction.plan.contacts=kinematic\n", encoding="utf-8")
    dropped = sandbox.run(PHYSVERIFY_ITER_ARM="it3-ok", DOCTOR_DROP_SETS="1")
    assert dropped.returncode == 2 and "does not list the arm's override" in dropped.stderr
    assert sandbox.commands() == ["doctor"]


def test_resume_with_another_arm_is_refused(sandbox: Sandbox) -> None:
    assert sandbox.run("1", "1", PHYSVERIFY_ITER_ARM="it3-r").returncode == 0
    result = sandbox.run("2", PHYSVERIFY_ITER_ARM="it3-ak")
    assert result.returncode == 2 and "belongs to CHAIN_ARM=it3-r" in result.stderr
