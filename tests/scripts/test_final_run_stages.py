"""scripts/h200/final_run.sh stage by stage against a fake physverify: the rehearsal transcript of
stages 0-8, the scores and comparisons of stages 9-10, the sealed versions of the preflight, and
the iteration stage 1 writes as the one stage 2 adopts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from repo_configs import CALIBRATION_FILE, copy_repo_configs, declare
from shell_sandbox import Sandbox, executable
from test_perception_iteration_script import IterationSandbox

from physverify.calibration.settings import calibration_settings
from physverify.config.loader import load_settings

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "h200" / "final_run.sh"
ITERATION_SCRIPT = REPO / "scripts" / "h200" / "run_perception_iteration.sh"
SCRIPT_NAMES = ("_common.sh", "_steps.sh", "_guards.sh", "final_run.sh")
TOOLS = (
    "awk", "basename", "bash", "cat", "cut", "dirname", "env", "find", "grep", "head", "mkdir",
    "mv", "nproc", "realpath", "rm", "sed", "sha256sum", "tee", "touch",
)  # fmt: skip
HEAD = "c0ffee1234567" + "0" * 27
PREFIX = re.compile(r"--profile precise --device cpu --set runtime\.threads=\d+ ")
RUNNERS = ("run_perception_iteration.sh", "run_freeze.sh", "run_human_agreement.sh")
# A physverify call is logged as "pv [PHYSVERIFY_UNSEAL=...] ARGS" and prints what final_run.sh
# reads back; the interpreter's other uses (the preflight's development-set check, the configured
# version) run for real. freeze_query.py answers a first look on an empty ledger.
FAKE_PYTHON = r"""#!/bin/bash
if [[ "$1" == - || ("$1" == -c && "$2" != *physverify.cli.main*) ]]; then
    exec "$REAL_PYTHON" "$@"
fi
if [[ "$1" != -c ]]; then
    printf 'query %s %s\n' "$(basename "$1")" "${*:2}" >> "$CALLS"
    [[ "$2" != look-state ]] || echo first
    exit 0
fi
shift 2
args="$*"
printf 'pv %s%s\n' "${PHYSVERIFY_UNSEAL+PHYSVERIFY_UNSEAL=$PHYSVERIFY_UNSEAL }" "$args" >> "$CALLS"
option() { sed -n "s/.*--$1 \([^ ]*\).*/\1/p" <<<"$args"; }
new_run() {
    local n
    n=$(( $(cat "$PHYSVERIFY__PATHS__RUNS_DIR/counter" 2>/dev/null || echo 0) + 1 ))
    RUN="$PHYSVERIFY__PATHS__RUNS_DIR/r$n"
    mkdir -p "$RUN"
    echo "$n" > "$PHYSVERIFY__PATHS__RUNS_DIR/counter"
    echo "run: $RUN"
}
case "$args" in
  *" calibrate evaluate "*)
    new_run; echo '{}' > "$RUN/evaluation.json"; echo "evaluation: $RUN/evaluation.json" ;;
  *" diagnose attribution "*) new_run; echo '{}' > "$RUN/attribution.json" ;;
  *" selective-risk curve"*) new_run; echo '{}' > "$RUN/abstention_report.json" ;;
  *" baselines compare "*)
    part="$(option part)"
    [[ " ${COMPARE_FAILS:-} " != *" $part "* ]] || exit 1
    mkdir -p "$(option out)"
    echo '{}' > "$(option out)/$part.json"
    echo "cuts: $(option out)/cuts.json"
    echo "comparison: $(option out)/$part.json" ;;
  *" final-table comparison "*)
    echo '#' > "$(option out)/comparison_table.md"
    echo "comparison table: $(option out)/comparison_table.md" ;;
esac
exit 0
"""
FAKE_GIT = """#!/bin/bash
case "$*" in
  *" rev-parse HEAD") echo "$HEAD_SHA" ;;
esac
"""
# Stages 1, 2 and 7 of a real run: each runner logs its call and keeps its environment; the
# agreement study's test phase records its verify run, whose items.csv final_run.sh saves.
FAKE_RUNNER = """#!/bin/bash
printf 'script %s %s\\n' "$(basename "$0")" "$*" >> "$CALLS"
env > "$CALLS.$(basename "$0")"
if [[ "${1:-}" == test ]]; then
    h2="$PHYSVERIFY_AGREEMENT_DIR"
    mkdir -p "$h2/runs/r1/agreement"
    echo clip_id > "$h2/runs/r1/agreement/items.csv"
    printf 'TEST_VERIFY_RUN=%s\\nTEST_COMPARED=1\\n' "$h2/runs/r1" >> "$h2/state.env"
fi
"""
# The transcript of 43b9b03 but for the curve, whose sets were --set calibration.* there.
REHEARSAL_0_TO_8 = [
    "final-table preflight --artifact $CAL/calibration.json --out $F/table/preflight.json",
    "--set kubric_fresh.version=v2-dev-b3 extract run --dataset kubric_fresh --subset eval",
    "extract run --dataset physics_iq",
    "extract run --dataset pisa_real",
    (
        "PHYSVERIFY_UNSEAL=kubric_fresh --set kubric_fresh.version=v2-dev-b3 calibrate evaluate"
        " --artifact $CAL/calibration.json --dataset kubric_fresh --subset eval --no-real"
    ),
    (
        "final-table inputs --evaluation $F/runs/r1/evaluation.json"
        " --preflight $F/table/preflight.json"
    ),
    (
        "PHYSVERIFY_UNSEAL=physics_iq,pisa_real calibrate evaluate --artifact $CAL/calibration.json"
        " --dataset kubric_synthetic --subset calib --real-dataset physics_iq"
        " --real-dataset pisa_real"
    ),
    (
        "final-table inputs --evaluation $F/runs/r2/evaluation.json"
        " --preflight $F/table/preflight.json"
    ),
    (
        "PHYSVERIFY_UNSEAL=kubric_fresh --set kubric_fresh.version=v2-dev-b3 diagnose attribution"
        " --dataset kubric_fresh --subset eval"
    ),
    (
        "PHYSVERIFY_UNSEAL=kubric_fresh,physics_iq,pisa_real --set kubric_fresh.version=v2-dev-b3"
        " selective-risk curve --dataset kubric_fresh --subset eval --real-dataset physics_iq"
        " --real-dataset pisa_real"
    ),
    (
        "final-table build --out $F/table --kubric $F/runs/r1/evaluation.json"
        " --real $F/runs/r2/evaluation.json --curve $F/runs/r4/abstention_report.json"
        " --agreement-items $ITEMS --preflight $F/table/preflight.json"
        " --attribution $F/runs/r3/attribution.json --dry-run"
    ),
]
REHEARSAL_9_TO_10 = [
    (
        "--set kubric_fresh.version=v2-dev-b3 baselines score --judge autoeval"
        " --dataset kubric_fresh --subset eval --limit 5 --scores $F/vlm/scores"
        " --autoeval-root $AUTOEVAL"
    ),
    (
        "baselines score --judge autoeval --dataset physics_iq --limit 5 --scores $F/vlm/scores"
        " --autoeval-root $AUTOEVAL"
    ),
    (
        "baselines score --judge autoeval --dataset pisa_real --limit 5 --scores $F/vlm/scores"
        " --autoeval-root $AUTOEVAL"
    ),
    (
        "PHYSVERIFY_UNSEAL=kubric_fresh --set kubric_fresh.version=v2-dev-b3 baselines compare"
        " --part kubric --artifact $CAL/calibration.json --dataset kubric_fresh --subset eval"
        " --scores $F/vlm/scores --out $F/vlm"
    ),
    (
        "PHYSVERIFY_UNSEAL=physics_iq,pisa_real baselines compare --part real"
        " --artifact $CAL/calibration.json --real-dataset physics_iq --real-dataset pisa_real"
        " --scores $F/vlm/scores --out $F/vlm"
    ),
    (
        "final-table comparison --out $F/table --kubric $F/vlm/kubric.json --real $F/vlm/real.json"
        " --agreement-items $ITEMS --final-table $F/table/final_table.json"
    ),
]
SEALED = ("v3-test", "v3-test-r3b", "v2-dev-b5")


class FinalRunSandbox(Sandbox):
    """final_run.sh in a fake checkout of the committed configs, at HEAD, on cpu."""

    def __init__(self, root: Path) -> None:
        super().__init__(root.resolve(), SCRIPT_NAMES, TOOLS)
        self.configs = copy_repo_configs(self.repo / "configs")
        executable(self.fakes / "python", FAKE_PYTHON)
        executable(self.fakes / "git", FAKE_GIT)
        for name in RUNNERS:
            executable(self.scripts / name, FAKE_RUNNER)
        self.final = self.root / "final-v3"
        self.env.update(
            {
                "REAL_PYTHON": sys.executable,
                "HEAD_SHA": HEAD,
                "PHYSVERIFY_H200_ROOT": str(self.root),
                "PHYSVERIFY_H200_ENV": str(self.root / "no-env.sh"),
                "PHYSVERIFY_FINAL_EXPECT_COMMIT": HEAD[:7],
                "PHYSVERIFY_FINAL_EARLY_EXTRACTION_COMMIT": "e" * 40,
                "PHYSVERIFY_FINAL_CACHE_DIR": str(self.root / "cache"),
                "PHYSVERIFY_FINAL_PYTHON": str(self.fakes / "python"),
                "PHYSVERIFY_FINAL_AUTOEVAL_ROOT": str(self.root / "autoeval"),
                "PV_DEVICE": "cpu",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )

    def rehearse(self) -> None:
        """A dry run: development data, its own ledger and directory under scratch."""
        scratch = self.root / "scratch"
        calibration = self.root / "cal"
        calibration.mkdir()
        (calibration / "abstention.json").write_text("{}", encoding="utf-8")
        items = self.root / "h2_dev" / "items.csv"
        items.parent.mkdir()
        items.write_text("clip_id\n", encoding="utf-8")
        self.final = scratch / "final"
        self.env.update(
            {
                "PHYSVERIFY_FINAL_DRYRUN": "1",
                "PHYSVERIFY_FINAL_DIR": str(self.final),
                "PHYSVERIFY_FINAL_DRY_CALIBRATION": str(calibration),
                "PHYSVERIFY_FINAL_DRY_ITEMS": str(items),
                "PHYSVERIFY_FINAL_DRY_VLM_LIMIT": "5",
                "PHYSVERIFY__SEALED__LEDGER": str(scratch / "ledger.jsonl"),
            }
        )

    def preflight_done(self, device: str = "cpu") -> None:
        """Stage 0 of a real run recorded as passed (its pinned plans are of the real render)."""
        run = self.final / "run"
        run.mkdir(parents=True)
        (run / "steps.tsv").write_text("0\tpreflight\t0\t0\t-\n", encoding="utf-8")
        state = f"RUN_COMMIT={HEAD}\nRUN_DEVICE={device}\n"
        (run / "state.env").write_text(state, encoding="utf-8")
        (self.final / "calibration").mkdir()

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call("final_run.sh", *args, **env)

    def ok(self, *args: str, **env: str) -> None:
        result = self.run(*args, **env)
        assert result.returncode == 0, result.stdout + result.stderr

    def commands(self) -> list[str]:
        """The pv calls, with the sandbox's paths named as final_run.sh names them."""
        env = self.env
        names = {
            str(self.final): "$F",
            env.get("PHYSVERIFY_FINAL_DRY_CALIBRATION", f"{self.final}/calibration"): "$CAL",
            env.get(
                "PHYSVERIFY_FINAL_DRY_ITEMS", f"{self.final}/h2/runs/r1/agreement/items.csv"
            ): "$ITEMS",
            env["PHYSVERIFY_FINAL_AUTOEVAL_ROOT"]: "$AUTOEVAL",
        }
        found = []
        for line in self.lines():
            if line.startswith("pv "):
                for path in sorted(names, key=len, reverse=True):
                    line = line.replace(path, names[path])
                found.append(PREFIX.sub("", line[3:]))
        return found

    def state(self) -> str:
        return (self.final / "run" / "state.env").read_text(encoding="utf-8")

    def runner_env(self, runner: str, prefix: str) -> dict[str, str]:
        """The variables named ``prefix``* that ``runner`` last ran with."""
        text = Path(f"{self.calls}.{runner}").read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in text.splitlines() if line.startswith(prefix))


@pytest.fixture
def sandbox(tmp_path: Path) -> FinalRunSandbox:
    return FinalRunSandbox(tmp_path)


@pytest.fixture
def rehearsal(sandbox: FinalRunSandbox) -> FinalRunSandbox:
    sandbox.rehearse()
    return sandbox


def test_the_rehearsal_of_stages_0_to_8_keeps_its_transcript(rehearsal: FinalRunSandbox) -> None:
    rehearsal.ok("0", "8")
    assert rehearsal.commands() == REHEARSAL_0_TO_8


def test_no_call_changes_the_calibration_section(rehearsal: FinalRunSandbox) -> None:
    """A --set moves the configuration hash, hence the freeze every look of the run shares."""
    rehearsal.ok()
    assert not [c for c in rehearsal.commands() if "--set calibration." in c]


def test_stages_9_and_10_score_then_compare_then_tabulate(rehearsal: FinalRunSandbox) -> None:
    rehearsal.ok("0", "10")
    assert rehearsal.commands()[len(REHEARSAL_0_TO_8) :] == REHEARSAL_9_TO_10
    state = rehearsal.state()
    for key in ("VLM_KUBRIC", "VLM_REAL", "COMPARISON_TABLE"):
        assert f"\n{key}=" in state


def test_stage_10_needs_stages_0_to_9(rehearsal: FinalRunSandbox) -> None:
    result = rehearsal.run("10", "10")
    assert result.returncode == 2 and "step 0 has not completed" in result.stderr
    assert rehearsal.commands() == []


def test_a_resumed_stage_10_repeats_no_comparison(rehearsal: FinalRunSandbox) -> None:
    """The Kubric comparison saved before the real one failed is not looked at again."""
    failed = rehearsal.run("0", "10", COMPARE_FAILS="real")
    assert failed.returncode == 1
    done = len(rehearsal.commands())
    rehearsal.ok("10", "10")
    resumed = rehearsal.commands()[done:]
    assert [c for c in resumed if "--part kubric" in c] == []
    assert resumed == REHEARSAL_9_TO_10[-2:]
    rehearsal.ok("10", "10")
    assert len(rehearsal.commands()) == done + 2


def _as_real(command: str) -> str:
    """A rehearsal's call on the sets of the real run, whose counted version is the configured
    one (no --set)."""
    command = command.replace("--set kubric_fresh.version=v2-dev-b3 ", "")
    command = command.replace("physics_iq", "physics_iq_views")
    return command.replace("pisa_real", "objectron_static")


def _guarded(commands: list[str]) -> list[str]:
    return [c for c in commands if c.startswith("PHYSVERIFY_UNSEAL=")]


def test_the_guarded_calls_of_the_rehearsal_are_those_of_the_real_run(tmp_path: Path) -> None:
    """The real run also looks at each supplementary version, by --set, and at nothing else."""
    real = FinalRunSandbox(tmp_path / "real")
    real.preflight_done()
    real.ok("1", "10")
    rehearsal = FinalRunSandbox(tmp_path / "rehearsal")
    rehearsal.rehearse()
    rehearsal.ok("0", "10")
    looks = _guarded(real.commands())
    supplementary = [c for c in looks if "--set kubric_fresh.version=" in c]
    assert [c.split()[2] for c in supplementary] == [
        f"kubric_fresh.version={v}" for v in SEALED[1:]
    ]
    counted = [c for c in looks if c not in supplementary]
    assert counted == [_as_real(c) for c in _guarded(rehearsal.commands())]


def _assigned(name: str, text: str) -> list[str]:
    found = re.search(rf"^{name}=\(?([^)\n]*)\)?$", text, re.MULTILINE)
    assert found is not None, name
    return found.group(1).split()


def test_no_sealed_kubric_version_of_the_run_is_a_development_set() -> None:
    """A development set's look is exempt: none of the run's versions may be one."""
    text = SCRIPT.read_text(encoding="utf-8")
    versions = _assigned("COUNTED", text) + _assigned("SUPPLEMENTARY", text)
    assert tuple(versions) == SEALED
    pinned = re.findall(r"^    \[([a-z0-9-]+)\]=[0-9a-f]{64}$", text, re.MULTILINE)
    assert tuple(pinned) == SEALED
    settings = load_settings(REPO / "configs", environ={})
    declared = set(calibration_settings(settings).development_sets)
    assert not {f"kubric_fresh@{v}" for v in versions} & declared


@pytest.mark.parametrize("version", SEALED)
def test_the_preflight_stops_on_any_sealed_version_declared_for_development(
    rehearsal: FinalRunSandbox, version: str
) -> None:
    """Checked in a rehearsal too, though it reads development sets only."""
    text = (rehearsal.configs / CALIBRATION_FILE).read_text(encoding="utf-8")
    listed = re.search(r"(?ms)^development_sets = (\[.*?\])$", text)
    assert listed is not None
    declare(
        rehearsal.configs,
        CALIBRATION_FILE,
        "development_sets",
        listed.group(1).replace("[", f'["kubric_fresh@{version}", ', 1),
    )
    result = rehearsal.run("0", "0")
    assert result.returncode == 2 and "is a declared development set" in result.stderr


def test_the_final_iteration_id_passes_the_iteration_check() -> None:
    """Stage 1's id matches the pattern run_perception_iteration.sh enforces, matched by bash."""
    (iteration_id,) = _assigned("ITERATION_ID", SCRIPT.read_text(encoding="utf-8"))
    text = ITERATION_SCRIPT.read_text(encoding="utf-8")
    pattern = re.search(r"^ITERATION_ID_PATTERN='([^']+)'$", text, re.MULTILINE)
    assert pattern is not None
    matched = subprocess.run(
        ["bash", "-c", '[[ "$1" =~ $2 ]]', "_", iteration_id, pattern.group(1)], check=False
    )
    assert matched.returncode == 0, f"{iteration_id} does not match {pattern.group(1)}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_the_freeze_adopts_the_directory_the_calibrate_stage_writes(
    tmp_path: Path, device: str
) -> None:
    """run_perception_iteration.sh, run on the ranges and with what stage 1 gives it, accepts each
    range on the run's device and writes the directory stage 2 hands run_freeze.sh, under the
    iteration id stage 2 names."""
    gpu = {"PV_DEVICE": device, "CUDA_VISIBLE_DEVICES": "0"}
    final = FinalRunSandbox(tmp_path / "final")
    final.preflight_done(device)
    final.ok("1", "2", **gpu)
    calibrate = final.runner_env("run_perception_iteration.sh", "PHYSVERIFY_ITERATION_")
    called = "script run_perception_iteration.sh "
    ranges = [x.removeprefix(called).split() for x in final.lines() if x.startswith(called)]
    assert ranges == [["1", "3"], ["4", "5"]]
    iteration = IterationSandbox(tmp_path / "iteration")
    for steps in ranges:
        ran = iteration.run(
            *steps, PHYSVERIFY_H200_ROOT=str(final.root), HEAD_SHA=HEAD, **gpu, **calibrate
        )
        assert ran.returncode == 0, ran.stderr
    freeze = final.runner_env("run_freeze.sh", "PHYSVERIFY_FREEZE_")
    adopted = Path(freeze["PHYSVERIFY_FREEZE_ITERATIONS"])
    text = (adopted / "state.env").read_text(encoding="utf-8")
    state = dict(line.split("=", 1) for line in text.splitlines())
    assert state["CHAIN_ITER_ID"] == freeze["PHYSVERIFY_FREEZE_ITERATION"]
    assert (adopted / "calibration.json").is_file()
