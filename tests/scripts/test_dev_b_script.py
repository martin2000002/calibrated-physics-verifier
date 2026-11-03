"""scripts/h200/run_dev_b.sh against a fake interpreter and ``git``: the B reading of an
iteration on the development sets, with the committed configuration only."""

from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from shell_sandbox import Sandbox, executable

SCRIPT_NAMES = ("_common.sh", "_steps.sh", "_guards.sh", "run_dev_b.sh", "perception_arms.tsv")
TOOLS = (
    "awk", "basename", "bash", "cat", "cp", "cut", "dirname", "env", "flock", "grep", "mkdir",
    "mv", "nproc", "realpath", "sed", "sha256sum", "tee", "touch",
)  # fmt: skip
PREFIX = re.compile(r"^--profile precise --device cpu --set runtime\.threads=16 ")
FRESH_CLIPS = ("f1_projectile-s000__none", "f2_bounce-s001__r4_resize_L1")
FRESH = "--dataset kubric_fresh --subset full " + " ".join(f"--clips {c}" for c in FRESH_CLIPS)
HDRI_SET = "--set dataset.version=v1-precise-hdri-dev"
HDRI = "--dataset kubric_synthetic --subset calib_clean"
TWO_FRESH = "v2-fresh v2-dev-b2"
# The development-version query answers from DECLARED_FRESH (the declared kubric_fresh versions,
# v2-fresh by default) and v1-precise-hdri-dev, as run_query.py would (QUERY_FAIL refuses); pv
# calls are logged with the paths and PV_* variables they see; doctor answers from the
# environment; the steps write what they would.
FAKE_PYTHON = r"""#!/bin/bash
CAL="$PHYSVERIFY__PATHS__CALIBRATION_DIR"
if [[ "$1" != -c ]]; then
    printf 'query %s\n' "${*:2}" >> "$CALLS"
    [[ -z "${QUERY_FAIL:-}" ]] || { echo "run_query: no version" >&2; exit 2; }
    args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do [[ "${args[i]}" == development-version ]] && break; done
    dataset="${args[i + 1]}"
    wanted="${args[i + 2]:-}"
    declared=(v1-precise-hdri-dev)
    [[ "$dataset" == kubric_fresh ]] && read -ra declared <<<"${DECLARED_FRESH:-v2-fresh}"
    if [[ -n "$wanted" ]]; then
        [[ " ${declared[*]} " == *" $wanted "* ]] \
            || { echo "run_query: does not declare $dataset@$wanted" >&2; exit 2; }
        echo "$wanted"
    elif [[ ${#declared[@]} -eq 1 ]]; then
        echo "${declared[0]}"
    else
        echo "run_query: declares ${#declared[@]} versions of $dataset, not one" >&2; exit 2
    fi
    exit 0
fi
shift 2
printf 'pv %s\n' "$*" >> "$CALLS"
printf 'env %s|%s|%s|%s|%s\n' "${CUDA_VISIBLE_DEVICES-unset}" "$PHYSVERIFY__PATHS__CACHE_DIR" \
    "$CAL" "$PHYSVERIFY__PATHS__RUNS_DIR" "$(compgen -e -X '!PV_*' | tr '\n' ' ')" >> "$CALLS"
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
        [[ "$arg" == *.version=* || ( -z "${DOCTOR_DROP_ARM:-}" && "$arg" == *=* ) ]] \
            && echo "source: --set $arg"
    done
    echo "profile: $2"
    echo "device: $4"
    echo "config_dir: $REPO_CONFIGS"
    echo "paths.cache_dir: $PHYSVERIFY__PATHS__CACHE_DIR (env)"
    echo "paths.runs_dir: $PHYSVERIFY__PATHS__RUNS_DIR"
    echo "paths.calibration_dir: ${DOCTOR_CALIBRATION_DIR:-$CAL}"
    echo "paths.error_profile: $CAL/stage_errors.precise.json" ;;
  *" extract run "*) exit "${CLIP_EXIT:-0}" ;;
  *" diagnose stages "*)
    new_run; mkdir -p "$CAL"; echo b-profile > "$CAL/stage_errors.precise.json" ;;
  *" calibrate stage-sigma "*) mkdir -p "$CAL"; echo sigma > "$CAL/stage_sigma.json" ;;
  *" calibrate dev-score "*)
    new_run
    mkdir -p "$(dirname "${!#}")"; echo '{"levels": []}' > "${!#}"
    echo "selection (perceived): n_met 2" ;;
  *" calibrate evaluate "*)
    new_run; echo '{}' > "$RUN/evaluation.json"; echo "evaluation: $RUN/evaluation.json" ;;
esac
exit 0
"""
FAKE_GIT = """#!/bin/bash
case "$*" in
  *" rev-parse HEAD") echo "c0ffee1234567" ;;
  *" status --porcelain "*) printf '%s' "${DIRTY_FILES:-}" ;;
esac
"""
FORBIDDEN = '#!/bin/bash\necho "forbidden $(basename "$0") $*" >> "$CALLS"\n'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_iteration(directory: Path, *, steps: int = 5) -> Path:
    """A completed run_perception_iteration.sh directory on cpu, precise."""
    (directory / "calibration").mkdir(parents=True)
    artifact, profile = b"calibration of A", b"profile of A"
    (directory / "calibration.json").write_bytes(artifact)
    (directory / "calibration" / "stage_errors.precise.json").write_bytes(profile)
    state = {
        "CHAIN_DEVICE": "cpu",
        "CHAIN_PROFILE": "precise",
        "DEVSCORE_RUN": str(directory / "runs" / "r1"),
        "CALIBRATION_SHA": _sha(artifact),
        "PROFILE_SHA": _sha(profile),
    }
    lines = "".join(f"{k}={shlex.quote(v)}\n" for k, v in state.items())
    (directory / "state.env").write_text(lines, encoding="utf-8")
    rows = "".join(f"{n}\tstep\t0\t9\t-\n" for n in range(1, steps + 1))
    (directory / "steps.tsv").write_text(rows, encoding="utf-8")
    return directory


class DevBSandbox(Sandbox):
    def __init__(self, root: Path) -> None:
        super().__init__(root.resolve(), SCRIPT_NAMES, TOOLS)
        self.iteration = make_iteration(self.root / "iterations" / "final-cpu")
        self.out = self.iteration / "dev-b"
        self.clips = self.root / "fresh_clips.txt"
        self.clips.write_text("\n".join(FRESH_CLIPS) + "\n", encoding="utf-8")
        (self.repo / "configs").mkdir()
        executable(self.fakes / "python", FAKE_PYTHON)
        executable(self.fakes / "git", FAKE_GIT)
        for name in ("uv", "nvidia-smi"):
            executable(self.fakes / name, FORBIDDEN)
        self.env.update(
            {
                "REPO_CONFIGS": str(self.repo / "configs"),
                "PHYSVERIFY_H200_ROOT": str(self.root),
                "PHYSVERIFY_H200_ENV": str(self.root / "no-env.sh"),
                "PHYSVERIFY_DEV_B_ITERATION_DIR": str(self.iteration),
                "PHYSVERIFY_DEV_B_EXPECT_COMMIT": "c0ffee1",
                "PHYSVERIFY_DEV_B_CACHE_DIR": str(self.root / "cache-v2-final"),
                "PHYSVERIFY_DEV_B_CLIPS_FILE": str(self.clips),
                "PHYSVERIFY_DEV_B_PYTHON": str(self.fakes / "python"),
            }
        )

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call("run_dev_b.sh", *args, **env)

    def commands(self) -> list[str]:
        return [PREFIX.sub("", line[3:]) for line in self.lines() if line.startswith("pv ")]

    def envs(self) -> list[list[str]]:
        return [line[4:].split("|") for line in self.lines() if line.startswith("env ")]


@pytest.fixture
def sandbox(tmp_path: Path) -> DevBSandbox:
    return DevBSandbox(tmp_path)


def _expected(sandbox: DevBSandbox, fresh_version: str = "v2-fresh") -> list[str]:
    devscore = sandbox.out / "fresh" / "devscore.json"
    artifact = sandbox.iteration / "calibration.json"
    fresh_set = f"--set kubric_fresh.version={fresh_version}"
    return [
        f"{HDRI_SET} doctor",
        f"{fresh_set} doctor",
        f"{fresh_set} diagnose stages {FRESH} --write-profile",
        f"{fresh_set} extract run --strict-profile {FRESH}",
        f"{fresh_set} calibrate stage-sigma {FRESH}",
        f"{fresh_set} calibrate dev-score {FRESH} --level both --out {devscore}",
        f"{HDRI_SET} extract run --strict-profile {HDRI}",
        f"{HDRI_SET} calibrate evaluate --artifact {artifact} {HDRI} --no-real --score-all-clips",
    ]


def test_the_reading_runs_both_sets_with_their_own_paths(sandbox: DevBSandbox) -> None:
    before = {p: p.read_bytes() for p in (sandbox.iteration / "calibration").iterdir()}
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == _expected(sandbox)
    a_calibration = str(sandbox.iteration / "calibration")
    fresh = [str(sandbox.out / "fresh" / "calibration"), str(sandbox.out / "fresh" / "runs")]
    hdri = [a_calibration, str(sandbox.out / "hdri" / "runs")]
    envs = sandbox.envs()
    assert [env[2:4] for env in envs] == [hdri, fresh, fresh, fresh, fresh, fresh, hdri, hdri]
    assert all(env[1] == str(sandbox.root / "cache-v2-final") for env in envs)
    assert {p: p.read_bytes() for p in (sandbox.iteration / "calibration").iterdir()} == before
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "kubric_fresh@v2-fresh (2 clips" in summary and "n_met 2" in summary
    assert "every clean clip scored with" in summary


def test_a_declared_fresh_version_is_read_by_name(sandbox: DevBSandbox) -> None:
    """With two kubric_fresh versions declared, the one named is the one every fresh call reads."""
    result = sandbox.run(DECLARED_FRESH=TWO_FRESH, PHYSVERIFY_DEV_B_FRESH_VERSION="v2-dev-b2")
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == _expected(sandbox, "v2-dev-b2")
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "kubric_fresh@v2-dev-b2 (2 clips" in summary


def _record_arm(sandbox: DevBSandbox, arm: str | None) -> None:
    if arm is not None:
        with (sandbox.iteration / "state.env").open("a", encoding="utf-8") as state:
            state.write(f"CHAIN_ARM={arm}\n")


K_ARM = (
    "extraction.plan.depth=vda_large check_variants.R5a=los_tilt check_variants.R5b=los_ballistic "
    "verification.thresholds.R5a=13.28 verification.threshold_units.R5a=accel_tilt_chi2_4 "
    "calibration.tau_tolerance_units.R5a=accel_tilt_chi2_4"
)


def test_the_iteration_arm_goes_as_set_to_every_call(sandbox: DevBSandbox) -> None:
    """The arm the iteration ran (CHAIN_ARM) is replayed: each of its swaps follows the set's
    version on every call, so a reading of arm k never falls back to the committed depth."""
    _record_arm(sandbox, "k")
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    arm_sets = " ".join(f"--set {swap}" for swap in K_ARM.split())
    expected = [
        re.sub(r"(--set \S+\.version=\S+)", rf"\1 {arm_sets}", c) for c in _expected(sandbox)
    ]
    assert sandbox.commands() == expected
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert f"arm k ({arm_sets})" in summary


@pytest.mark.parametrize("arm", ["none", None])
def test_an_iteration_without_an_arm_adds_no_set(sandbox: DevBSandbox, arm: str | None) -> None:
    _record_arm(sandbox, arm)
    assert sandbox.run().returncode == 0
    assert sandbox.commands() == _expected(sandbox)


@pytest.mark.parametrize(
    ("arm", "env", "message"),
    [
        ("undeclared-x", {}, "not declared exactly once"),
        ("k", {"DOCTOR_DROP_ARM": "1"}, "does not list the override --set extraction.plan.depth"),
    ],
)
def test_an_undeclared_arm_or_a_doctor_without_its_sets_stops(
    sandbox: DevBSandbox, arm: str, env: dict[str, str], message: str
) -> None:
    _record_arm(sandbox, arm)
    result = sandbox.run(**env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert all(c.endswith("doctor") for c in sandbox.commands())


def test_no_scratch_variable_or_unseal_is_needed(sandbox: DevBSandbox) -> None:
    """I30e: no PV_* variable reaches physverify and nothing is unsealed or relabelled by hand."""
    assert sandbox.run().returncode == 0
    assert all(env[4].strip() == "" for env in sandbox.envs())
    every = " ".join(sandbox.lines())
    assert not re.search(r"UNSEAL|sealed\.sets|forbidden|--real ", every)
    source = (sandbox.scripts / "run_dev_b.sh").read_text(encoding="utf-8")
    assert not re.search(r"PV_FRESH_AS_CALIB|PV_EVAL_ALL|PV_OVERLAY|PHYSVERIFY_UNSEAL=", source)


def test_a_dry_run_prints_the_commands_and_writes_nothing(sandbox: DevBSandbox) -> None:
    result = sandbox.run(PHYSVERIFY_DEV_B_DRY_RUN="1")
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == _expected(sandbox)[:2]
    printed = [line for line in result.stdout.splitlines() if line.startswith("dry run")]
    shown = [PREFIX.sub("", line.split("physverify ", 1)[1]) for line in printed]
    assert shown == _expected(sandbox)[2:]
    assert not sandbox.out.exists()


def _break_artifact(sandbox: DevBSandbox) -> None:
    (sandbox.iteration / "calibration.json").write_bytes(b"another fit")


def _break_profile(sandbox: DevBSandbox) -> None:
    (sandbox.iteration / "calibration" / "stage_errors.precise.json").write_bytes(b"other")


def _drop_step(sandbox: DevBSandbox) -> None:
    (sandbox.iteration / "steps.tsv").write_text("1\tstep\t0\t9\t-\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("breaker", "message"),
    [
        (_break_artifact, "is not the one"),
        (_break_profile, "is not the error profile"),
        (_drop_step, "has not completed its step 2"),
    ],
)
def test_an_iteration_that_is_not_the_recorded_one_is_refused(
    sandbox: DevBSandbox, breaker: Callable[[DevBSandbox], None], message: str
) -> None:
    breaker(sandbox)
    result = sandbox.run()
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not any(c.startswith(("diagnose", "extract", "calibrate")) for c in sandbox.commands())


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_UNSEAL": "kubric_fresh"}, "PHYSVERIFY_UNSEAL is set"),
        ({"PHYSVERIFY__DATASET__VERSION": "v1-precise-hdri-dev"}, "PHYSVERIFY__DATASET__VERSION"),
        ({"PHYSVERIFY__SEALED__SETS": "[]"}, "PHYSVERIFY__SEALED__SETS"),
        ({"PHYSVERIFY_DEV_B_ITERATION_DIR": ""}, "PHYSVERIFY_DEV_B_ITERATION_DIR"),
        ({"PHYSVERIFY_DEV_B_EXPECT_COMMIT": "deadbeef"}, "expected deadbeef"),
        ({"DIRTY_FILES": " M configs/verifier/calibration.toml\n"}, "uncommitted"),
        ({"PHYSVERIFY_DEV_B_CLIPS_FILE": ""}, "PHYSVERIFY_DEV_B_CLIPS_FILE"),
        ({"PHYSVERIFY_DEV_B_DRY_RUN": "yes"}, "PHYSVERIFY_DEV_B_DRY_RUN"),
        ({"QUERY_FAIL": "1"}, "run_query: no version"),
        ({"DECLARED_FRESH": TWO_FRESH}, "versions of kubric_fresh, not one"),
        ({"PHYSVERIFY_DEV_B_FRESH_VERSION": "v3"}, "does not declare kubric_fresh@v3"),
        ({"PHYSVERIFY_DEV_B_FRESH_VERSION": "v2 x"}, "is not a version"),
        ({"DOCTOR_CALIBRATION_DIR": "/elsewhere"}, "paths.calibration_dir"),
    ],
)
def test_a_startup_guard_stops_before_any_step(
    sandbox: DevBSandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert all(c.endswith("doctor") for c in sandbox.commands())


def test_a_clip_list_with_a_non_id_is_refused(sandbox: DevBSandbox) -> None:
    sandbox.clips.write_text("f1_projectile-s000__none\n$(touch x)\n", encoding="utf-8")
    result = sandbox.run()
    assert result.returncode == 2 and "is not a clip id" in result.stderr
    assert sandbox.commands() == []


@pytest.mark.parametrize("inside", ["calibration", "calibration/b"])
def test_an_output_in_the_iteration_calibration_is_refused(
    sandbox: DevBSandbox, inside: str
) -> None:
    result = sandbox.run(PHYSVERIFY_DEV_B_DIR=str(sandbox.iteration / inside))
    assert result.returncode == 2 and "never writes there" in result.stderr, result.stderr


def test_resume_needs_the_earlier_steps_and_the_same_clips(sandbox: DevBSandbox) -> None:
    early = sandbox.run("2")
    assert early.returncode == 2 and "step 1 has not completed" in early.stderr
    assert sandbox.run("1", "1").returncode == 0
    sandbox.calls.unlink()
    assert sandbox.run("2").returncode == 0
    assert sandbox.commands() == _expected(sandbox)[:2] + _expected(sandbox)[3:]
    sandbox.clips.write_text(FRESH_CLIPS[0] + "\n", encoding="utf-8")
    other = sandbox.run("5")
    assert other.returncode == 2 and "belongs to DEV_B_CLIPS_SHA" in other.stderr
