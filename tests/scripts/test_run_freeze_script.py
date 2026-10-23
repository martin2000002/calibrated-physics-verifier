"""scripts/h200/run_freeze.sh against a fake interpreter and ``git``."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from shell_sandbox import Sandbox, executable

REPO = Path(__file__).resolve().parents[2]
H200 = REPO / "scripts" / "h200"
SCRIPT_NAMES = (
    "_common.sh",
    "_steps.sh",
    "_guards.sh",
    "run_freeze.sh",
    "run_query.py",
    "freeze_query.py",
)
TOOLS = (
    "awk", "basename", "bash", "cat", "cmp", "cp", "cut", "dirname", "env", "flock", "grep",
    "mkdir", "mv", "nproc", "realpath", "sed", "sha256sum", "tee", "touch",
)  # fmt: skip
HEAD = "c0ffee1234567" + "0" * 27
PREFIX = re.compile(
    r"^--profile precise --device cpu --set runtime\.threads=32 "
    r"(--set extraction\.plan\.detect=grounding_dino_tiny )?"
)
RECORD = "e" * 64
CLIPS = "--clips k-calib-1 --clips k-calib-2"
KS = "--dataset kubric_synthetic --subset full"
# freeze_query.py select and chosen run for real (they only read the iteration directories);
# diff, manifest and artifacts answer from the environment. pv calls are logged with the paths
# and the unseal variable they see.
FAKE_PYTHON = r"""#!/bin/bash
CAL="$PHYSVERIFY__PATHS__CALIBRATION_DIR"
if [[ "$1" != -c ]]; then
    script="$(basename "$1")"
    printf 'query %s %s\n' "$script" "${*:2}" >> "$CALLS"
    case "$script $2" in
      "freeze_query.py select" | "freeze_query.py chosen") exec "$REAL_PYTHON" "$@" ;;
      "freeze_query.py diff") printf '%s' "${J3_CHANGES:-}" ;;
      "freeze_query.py manifest")
        echo '{"freeze": 1}' > "${!#}"; echo "freeze: ${!#} sha256 f00" ;;
      "freeze_query.py artifacts") exit "${ARTIFACTS_EXIT:-0}" ;;
      "run_query.py evaluation") echo "real sets physics_iq, pisa_real; h1_primary present" ;;
    esac
    exit 0
fi
shift 2
printf 'pv %s\n' "$*" >> "$CALLS"
printf 'env %s|%s|%s|%s|%s|%s\n' "${CUDA_VISIBLE_DEVICES-unset}" "${PHYSVERIFY_UNSEAL-unset}" \
    "$PHYSVERIFY__PATHS__CACHE_DIR" "$CAL" "$PHYSVERIFY__PATHS__RUNS_DIR" \
    "$PHYSVERIFY__RUNTIME__HEAVY_LOCK" >> "$CALLS"
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
    echo "profile: $2"
    echo "device: ${DOCTOR_DEVICE:-$4}"
    echo "config_dir: $REPO_CONFIGS"
    echo "paths.cache_dir: $PHYSVERIFY__PATHS__CACHE_DIR (env)"
    echo "paths.runs_dir: $PHYSVERIFY__PATHS__RUNS_DIR"
    echo "paths.calibration_dir: $CAL"
    echo "paths.error_profile: ${DOCTOR_ERROR_PROFILE:-$CAL/stage_errors.precise.json}"
    echo "paths.thresholds: $CAL/thresholds.json" ;;
  *" extract run "*) exit "${CLIP_EXIT:-0}" ;;
  *" calibrate evaluate "*)
    new_run; echo '{}' > "$RUN/evaluation.json"
    [[ -z "${SKIPPED:-}" ]] || echo "real dataset pisa_real skipped: no extracted clip"
    echo "evaluation: $RUN/evaluation.json" ;;
  *" exclusions r2-clips "*) printf 'k-calib-1\nk-calib-2\n' > "${!#}" ;;
  *" verify "*)
    new_run; hash=h
    [[ "$*" != *grounding_dino_tiny* ]] || hash="${TINY_HASH:-h}"
    echo "$hash" > "$RUN/checks_hash.txt" ;;
  *" exclusions decide "*) echo "decided" ;;
  *" exclusions show") echo "record: $CAL/exclusions.json sha256 ${RECORD:-}" ;;
  *" selective-risk fit") echo '{}' > "$CAL/abstention.json" ;;
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
ADOPTED = ("calibration.json", "stage_errors.precise.json", "stage_sigma.json")


class FreezeSandbox(Sandbox):
    def __init__(self, root: Path, iteration_factory: Callable[..., Path]) -> None:
        super().__init__(root.resolve(), SCRIPT_NAMES, TOOLS)
        self.final = self.root / "final"
        self.out = self.final / "run-freeze"
        (self.repo / "configs").mkdir()
        executable(self.fakes / "python", FAKE_PYTHON)
        executable(self.fakes / "git", FAKE_GIT)
        for name in ("uv", "nvidia-smi"):
            executable(self.fakes / name, FORBIDDEN)
        # PHYSVERIFY_FREEZE_ITERATIONS is split on spaces, as run_perception_iteration.sh names
        # them.
        made = [
            iteration_factory(self.root / "iterations", name, n_met, auroc, commit=HEAD)
            for name, n_met, auroc in (("it6-fixes", 2, 0.9), ("it7-state", 3, 0.6))
        ]
        other, self.chosen = (d.rename(d.with_name(d.name.replace(" ", "-"))) for d in made)
        v1 = self.root / "v1" / "calibration.json"
        v1.parent.mkdir()
        v1.write_text("v1", encoding="utf-8")
        self.env.update(
            {
                "REAL_PYTHON": sys.executable,
                "REPO_CONFIGS": str(self.repo / "configs"),
                "PHYSVERIFY_H200_ROOT": str(self.root),
                "PHYSVERIFY_H200_ENV": str(self.root / "no-env.sh"),
                "HEAD_SHA": HEAD,
                "RECORD": RECORD,
                "PHYSVERIFY_FREEZE_EXPECT_COMMIT": "c0ffee1",
                "PHYSVERIFY_FREEZE_ITERATION": "it7-state",
                "PHYSVERIFY_FREEZE_ITERATIONS": f"{other} {self.chosen}",
                "PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT": "b" * 40,
                "PHYSVERIFY_FREEZE_V1_COMMIT": "a" * 40,
                "PHYSVERIFY_FREEZE_V1_CALIBRATION": str(v1),
                "PHYSVERIFY_FREEZE_CACHE_DIR": str(self.root / "cache"),
                "PHYSVERIFY_FREEZE_PYTHON": str(self.fakes / "python"),
            }
        )

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call("run_freeze.sh", *args, **env)

    def pv(self) -> list[str]:
        return [line[3:] for line in self.lines() if line.startswith("pv ")]

    def commands(self) -> list[str]:
        return [PREFIX.sub("", line) for line in self.pv()]

    def queries(self) -> list[str]:
        return [line[6:] for line in self.lines() if line.startswith("query ")]

    def state(self) -> dict[str, str]:
        text = (self.out / "state.env").read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in text.splitlines())


@pytest.fixture
def sandbox(tmp_path: Path, iteration_factory: Callable[..., Path]) -> FreezeSandbox:
    return FreezeSandbox(tmp_path, iteration_factory)


def _expected(final: Path) -> list[str]:
    runs = final / "runs"
    return [
        "doctor",
        "extract run --dataset physics_iq --subset full",
        "extract run --dataset pisa_real --subset full",
        "extract run --dataset kubric_synthetic --subset calib",
        "calibrate evaluate --dataset kubric_synthetic --subset calib --real",
        f"exclusions r2-clips --out {final / 'r2_clips.txt'}",
        f"verify {KS} --level perceived --policy raw {CLIPS}",
        f"extract run {KS} {CLIPS}",
        f"verify {KS} --level perceived --policy raw {CLIPS}",
        (
            f"exclusions decide --evaluation {runs / 'r1' / 'evaluation.json'} --r2-base-run r2 "
            "--r2-tiny-run r3"
        ),
        "exclusions show",
        "selective-risk fit",
    ]


def test_the_script_parses_and_is_executable() -> None:
    subprocess.run(["bash", "-n", str(H200 / "run_freeze.sh")], check=True)
    assert os.access(H200 / "run_freeze.sh", os.X_OK)


def test_the_freeze_adopts_the_chosen_iteration_and_writes_the_manifest(
    sandbox: FreezeSandbox,
) -> None:
    """The sequence, cpu and precise, every path in $F, no look at all."""
    result = sandbox.run()
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == _expected(sandbox.final)
    assert all(PREFIX.match(line) for line in sandbox.pv())
    every = " ".join(sandbox.lines())
    assert not re.search(r"--subset eval|kubric_fresh|_views|objectron|curve|forbidden", every)
    final = str(sandbox.final)
    for line in (x for x in sandbox.lines() if x.startswith("env ")):
        cuda, unseal, cache, *mine = line[4:].split("|")
        assert (cuda, unseal, cache) == ("", "unset", str(sandbox.root / "cache"))
        assert all(path.startswith(final + "/") for path in mine)
    for name in ADOPTED:
        source = sandbox.chosen / ("" if name == "calibration.json" else "calibration") / name
        assert (sandbox.final / "calibration" / name).read_bytes() == source.read_bytes()
    queries = sandbox.queries()
    assert f"freeze_query.py diff j3 {HEAD}" in queries
    (manifest,) = [q for q in queries if q.startswith("freeze_query.py manifest ")]
    assert f"--record-sha {RECORD} --p2-commit {'b' * 40} --v1-commit {'a' * 40}" in manifest
    assert manifest.endswith(f"--out {sandbox.final / 'freeze.json'}")
    assert queries[-1] == f"freeze_query.py artifacts {sandbox.final / 'freeze.json'}"
    state = sandbox.state()
    assert state["ADOPTED"] == "it7-state" and state["RECORD_SHA"] == RECORD
    assert state["FREEZE_COMMIT"] == HEAD and state["R2_TINY_RUN"] == "r3"
    summary = (sandbox.out / "summary.md").read_text(encoding="utf-8")
    assert "h1_primary present" in summary and RECORD in summary and "it7-state" in summary


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_FREEZE_EXPECT_COMMIT": ""}, "PHYSVERIFY_FREEZE_EXPECT_COMMIT"),
        ({"HEAD_SHA": "deadbeef0000"}, "expected c0ffee1"),
        ({"DIRTY_FILES": "?? configs/local.toml\n"}, "uncommitted or untracked"),
        ({"PHYSVERIFY_UNSEAL": "kubric_synthetic:eval"}, "PHYSVERIFY_UNSEAL is set"),
        ({"PHYSVERIFY_UNSEAL_REPEAT": "1"}, "PHYSVERIFY_UNSEAL_REPEAT is set"),
        ({"PHYSVERIFY__SEALED__SETS": "[]"}, "PHYSVERIFY__SEALED__SETS"),
        ({"PHYSVERIFY_CONFIG_DIR": "/other"}, "PHYSVERIFY_CONFIG_DIR"),
        ({"PHYSVERIFY_FREEZE_ITERATION": ""}, "PHYSVERIFY_FREEZE_ITERATION"),
        (
            {"PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT": ""},
            "PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT",
        ),
        ({"PHYSVERIFY_FREEZE_V1_CALIBRATION": "/no/such.json"}, "no v1 calibration"),
        ({"PHYSVERIFY_FREEZE_THREADS": "0"}, "PHYSVERIFY_FREEZE_THREADS"),
        ({"DOCTOR_DEVICE": "cuda:0"}, "not cpu"),
        ({"DOCTOR_ERROR_PROFILE": "/elsewhere/stage_errors.precise.json"}, "outside"),
    ],
)
def test_a_startup_guard_stops_before_any_step(
    sandbox: FreezeSandbox, env: dict[str, str], message: str
) -> None:
    """A refused freeze: nothing runs, nothing is adopted."""
    result = sandbox.run(**env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() in ([], ["doctor"])
    assert not (sandbox.final / "calibration").exists()


@pytest.mark.parametrize("target", ["a8v2", "iterations/final", "kubric-data", "kubric-data/runs"])
def test_a_freeze_directory_near_other_artifacts_is_refused(
    sandbox: FreezeSandbox, target: str
) -> None:
    result = sandbox.run(PHYSVERIFY_FREEZE_DIR=str(sandbox.root / target))
    assert result.returncode == 2 and "never writes there" in result.stderr, result.stderr
    assert sandbox.pv() == []


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"PHYSVERIFY_FREEZE_ITERATION": "it6-fixes"}, "the selection rule chooses it7-state"),
        ({"J3_CHANGES": "src/physverify/checks/r1.py\n"}, "refit needed"),
    ],
)
def test_the_wrong_choice_or_a_refit_change_adopts_nothing(
    sandbox: FreezeSandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert sandbox.commands() == ["doctor"]
    assert not (sandbox.final / "calibration" / "calibration.json").exists()


def test_a_different_artifact_already_frozen_is_never_mixed(sandbox: FreezeSandbox) -> None:
    calibration = sandbox.final / "calibration"
    calibration.mkdir(parents=True)
    (calibration / "stage_sigma.json").write_text("another sigma", encoding="utf-8")
    result = sandbox.run()
    assert result.returncode == 2 and "differs" in result.stderr, result.stderr
    assert sandbox.commands() == ["doctor"]
    assert "ADOPTED" not in sandbox.state()


def test_a_resume_needs_the_earlier_steps_and_the_same_commit(sandbox: FreezeSandbox) -> None:
    assert sandbox.run("5").returncode == 2
    assert sandbox.run("1", "3").returncode == 0
    moved = sandbox.run("4", HEAD_SHA="c0ffee1" + "9" * 33)
    assert moved.returncode == 2 and "belongs to FREEZE_COMMIT" in moved.stderr
    resumed = sandbox.run("4")
    assert resumed.returncode == 0, resumed.stderr
    assert sandbox.commands() == [
        *_expected(sandbox.final)[:5],
        "doctor",
        *_expected(sandbox.final)[5:],
    ]


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"SKIPPED": "1"}, "a real dataset was skipped"),
        ({"TINY_HASH": "other"}, "different check parameters"),
        ({"RECORD": ""}, "no record sha256"),
        ({"ARTIFACTS_EXIT": "2"}, "failed (2)"),
    ],
)
def test_a_failed_check_stops_the_freeze_before_the_manifest_is_trusted(
    sandbox: FreezeSandbox, env: dict[str, str], message: str
) -> None:
    result = sandbox.run(**env)
    assert result.returncode == 2 and message in result.stderr, result.stderr
    assert "FREEZE_SHA" not in sandbox.state()


def test_decide_replaces_only_when_asked(sandbox: FreezeSandbox) -> None:
    assert sandbox.run(PHYSVERIFY_FREEZE_DECIDE_REPLACE="1").returncode == 0
    (decide,) = [c for c in sandbox.commands() if c.startswith("exclusions decide")]
    assert decide.endswith("--r2-tiny-run r3 --replace")
