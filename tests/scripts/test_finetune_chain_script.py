"""scripts/h200/run_finetune_chain.sh against a fake interpreter and a fake GPU slot."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from shell_sandbox import CALL_TIMEOUT_S, Sandbox, executable

SCRIPT_NAMES = ("_common.sh", "_steps.sh", "run_finetune_chain.sh")
TOOLS = (
    "bash", "dirname", "grep", "nproc", "sed", "awk", "tail", "mkdir", "touch", "realpath",
    "cat", "mv", "env", "nice",
)  # fmt: skip
PV = "--profile precise --device cuda --set runtime.threads=16 "
FAKE_PYTHON = """#!/bin/bash
shift 2
printf 'gpu=%s %s\\n' "${CUDA_VISIBLE_DEVICES-unset}" "$*" >> "$CALLS"
[[ "$*" == *"compare-generators status"* ]] && printf '%s\\n' "${FAKE_STATUS:-}"
if [[ -n "${FAIL_ON:-}" && "$*" == *"$FAIL_ON"* ]]; then
    printf '%s\\n' "${FAIL_TEXT:-}" >&2
    exit "${FAIL_CODE:-2}"
fi
exit 0
"""
FAKE_SLOT = """#!/bin/bash
printf 'slot %s\\n' "$1" >> "$CALLS"
shift
CUDA_VISIBLE_DEVICES=2 "$@"
"""
STEPS = (
    ("score-train", "finetune score --split train"),
    ("score-dev", "finetune score --split dev"),
    ("pairs", "finetune pairs"),
    ("beta-grid", "finetune beta-grid"),
    ("train", "finetune train --resume"),
    ("select", "finetune select"),
    ("gate-autoeval", "compare-generators gate-autoeval"),
    ("judge-dev", "compare-generators judge --split dev"),
    ("freeze", "compare-generators freeze"),
    ("generate-base", "compare-generators generate --arm base"),
    ("generate-ft", "compare-generators generate --arm ft"),
    ("score-base", "compare-generators score --arm base"),
    ("score-ft", "compare-generators score --arm ft"),
    ("judge-eval", "compare-generators judge --split eval"),
    ("compare", "compare-generators compare"),
    ("demo", "demo scenes --from-select --out"),
)
CPU_STEPS = frozenset({"pairs", "freeze", "compare"})
FIRED = "error: pilot stop rule fired: {}; the generator comparison is not evaluable (...)"
PASSED = "  gate: passed"


class ChainSandbox(Sandbox):
    def __init__(self, root: Path) -> None:
        super().__init__(root, SCRIPT_NAMES, TOOLS)
        executable(self.fakes / "python", FAKE_PYTHON)
        executable(self.fakes / "slot", FAKE_SLOT)
        (root / "calibration").mkdir()
        self.run_dir = root / "finetune-chain"
        self.env.update(
            {
                "PHYSVERIFY_H200_ROOT": str(root),
                "PHYSVERIFY_H200_ENV": str(root / "no-env.sh"),
                "PHYSVERIFY_FT_CALIBRATION": str(root / "calibration"),
                "PHYSVERIFY_FT_PYTHON": str(self.fakes / "python"),
                "PHYSVERIFY_FT_GPU_SLOT": str(self.fakes / "slot"),
            }
        )

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return self.call("run_finetune_chain.sh", *args, **env)

    def commands(self) -> list[str]:
        """The physverify calls without the status reads, prefixed with their GPU."""
        return [
            line.replace(PV, "")
            for line in self.lines()
            if line.startswith("gpu=") and "status" not in line
        ]

    def exits(self) -> dict[str, str]:
        rows = (self.run_dir / "steps.tsv").read_text(encoding="utf-8").splitlines()
        return {row.split("\t")[1]: row.split("\t")[2] for row in rows}


@pytest.fixture
def sandbox(tmp_path: Path) -> ChainSandbox:
    return ChainSandbox(tmp_path)


def test_all_runs_every_step_in_order_with_gpu_steps_in_the_slot(sandbox: ChainSandbox) -> None:
    result = sandbox.run("all", FAKE_STATUS=PASSED)
    assert result.returncode == 0, result.stderr
    commands = sandbox.commands()
    expected = [f"gpu={'' if name in CPU_STEPS else '2'} {call}" for name, call in STEPS]
    assert [c.split(" --out")[0] for c in commands] == [e.split(" --out")[0] for e in expected]
    slots = [line for line in sandbox.lines() if line.startswith("slot ")]
    assert slots == [f"slot ft-{name}" for name, _ in STEPS if name not in CPU_STEPS]
    assert set(sandbox.exits().values()) == {"0"}


def test_a_rerun_of_a_finished_chain_calls_nothing(sandbox: ChainSandbox) -> None:
    assert sandbox.run("all", FAKE_STATUS=PASSED).returncode == 0
    sandbox.calls.unlink()
    result = sandbox.run("all", FAKE_STATUS=PASSED)
    assert result.returncode == 0, result.stderr
    assert sandbox.lines() == []


def test_a_stop_rule_waiting_for_the_review_exits_3_before_training(sandbox: ChainSandbox) -> None:
    reason = "no pilot review recorded: run 'finetune pilot-review'"
    result = sandbox.run("all", FAIL_ON="beta-grid", FAIL_TEXT=FIRED.format(reason))
    assert result.returncode == 3
    assert sandbox.commands()[-1] == "gpu=2 finetune beta-grid"
    assert sandbox.exits()["beta-grid"] == "3"


def test_a_stop_rule_fired_on_pairs_exits_4_and_scores_no_eval(sandbox: ChainSandbox) -> None:
    reasons = "pairs per scored prompt 0.0 < 0.3; no pilot review recorded"
    result = sandbox.run("all", FAIL_ON="beta-grid", FAIL_TEXT=FIRED.format(reasons))
    assert result.returncode == 4
    assert not any("compare-generators" in c for c in sandbox.commands())
    assert "not evaluable" in (sandbox.run_dir / "notes.md").read_text(encoding="utf-8")


def test_another_beta_grid_failure_keeps_its_code(sandbox: ChainSandbox) -> None:
    result = sandbox.run("all", FAIL_ON="beta-grid", FAIL_TEXT="error: CUDA out of memory")
    assert result.returncode == 2
    assert sandbox.exits()["beta-grid"] == "2"


def test_without_a_passed_gate_the_judges_are_skipped_and_compare_still_runs(
    sandbox: ChainSandbox,
) -> None:
    result = sandbox.run("all", FAKE_STATUS="  gate: not run")
    assert result.returncode == 0, result.stderr
    commands = sandbox.commands()
    assert not any("judge" in c for c in commands)
    assert "gpu= compare-generators compare" in commands
    exits = sandbox.exits()
    assert exits["judge-dev"] == "1" and exits["judge-eval"] == "1" and exits["compare"] == "0"


def test_overrides_and_the_deviation_reach_the_calls(sandbox: ChainSandbox) -> None:
    result = sandbox.run(
        "all",
        FAKE_STATUS=PASSED,
        PHYSVERIFY_FT_SET="finetune.scorer=graded",
        PHYSVERIFY_FT_STOP_RULE_DEVIATION="declared",
    )
    assert result.returncode == 0, result.stderr
    commands = sandbox.commands()
    assert all("--set finetune.scorer=graded" in line for line in sandbox.lines() if "gpu=" in line)
    gated = [c for c in commands if "--stop-rule-deviation declared" in c]
    assert [c.split("graded ")[1].split(" --stop")[0] for c in gated] == [
        "finetune beta-grid",
        "finetune train --resume",
        "compare-generators freeze",
        "compare-generators generate --arm base",
        "compare-generators generate --arm ft",
    ]


def test_a_rerun_with_other_overrides_is_refused(sandbox: ChainSandbox) -> None:
    assert sandbox.run("score-train").returncode == 0
    result = sandbox.run("score-dev", PHYSVERIFY_FT_SET="finetune.scorer=graded")
    assert result.returncode == 2
    assert "use another PHYSVERIFY_FT_DIR" in result.stderr


def test_a_step_before_its_predecessors_is_refused(sandbox: ChainSandbox) -> None:
    result = sandbox.run("score-base")
    assert result.returncode == 2
    assert "step 1 (score-train) has not completed" in result.stderr
    assert sandbox.commands() == []


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("no-calibration", "PHYSVERIFY_FT_CALIBRATION"),
        ("local-toml", "configs/local.toml"),
        ("unknown-step", "unknown step 'nope'"),
    ],
)
def test_refusals(sandbox: ChainSandbox, setup: str, message: str) -> None:
    args, env = ["all"], {}
    if setup == "no-calibration":
        env["PHYSVERIFY_FT_CALIBRATION"] = str(sandbox.root / "missing")
    elif setup == "local-toml":
        (sandbox.repo / "configs").mkdir()
        (sandbox.repo / "configs" / "local.toml").write_text("", encoding="utf-8")
    else:
        args = ["nope"]
    result = sandbox.run(*args, **env)
    assert result.returncode == 2
    assert message in result.stderr
    assert sandbox.commands() == []


def test_review_passes_its_arguments_without_a_gpu(sandbox: ChainSandbox) -> None:
    result = sandbox.run("review", "--better", "7", "--note", "looked at all ten")
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == ["gpu= finetune pilot-review --better 7 --note looked at all ten"]


@pytest.mark.parametrize(
    "reasons",
    [
        "fit pairs 53 < 54",
        "fit pairs 53 < 54; no pilot review recorded: run 'finetune pilot-review'",
    ],
)
def test_a_pair_count_short_of_its_minimum_is_not_evaluable(
    sandbox: ChainSandbox, reasons: str
) -> None:
    result = sandbox.run("all", FAIL_ON="beta-grid", FAIL_TEXT=FIRED.format(reasons))
    assert result.returncode == 4


def test_a_better_winner_shortfall_is_not_evaluable(sandbox: ChainSandbox) -> None:
    reason = "3 of 10 inspected pairs show a better winner (< 7)"
    result = sandbox.run("all", FAIL_ON="beta-grid", FAIL_TEXT=FIRED.format(reason))
    assert result.returncode == 4


def test_a_stale_scorer_stamp_keeps_the_usage_exit(sandbox: ChainSandbox) -> None:
    reason = "the pairs were built under another scorer stamp: run 'finetune pairs' again"
    result = sandbox.run("all", FAIL_ON="beta-grid", FAIL_TEXT=FIRED.format(reason))
    assert result.returncode == 2
    assert sandbox.exits()["beta-grid"] == "2"


def test_a_failed_optional_step_is_not_rerun(sandbox: ChainSandbox) -> None:
    first = sandbox.run("all", FAKE_STATUS=PASSED, FAIL_ON="gate-autoeval", FAIL_CODE="1")
    assert first.returncode == 0, first.stderr
    sandbox.calls.unlink()
    assert sandbox.run("all", FAKE_STATUS=PASSED).returncode == 0
    assert sandbox.lines() == []


def test_a_busy_lock_on_an_optional_step_stops_and_is_retried(sandbox: ChainSandbox) -> None:
    first = sandbox.run("all", FAIL_ON="gate-autoeval", FAIL_CODE="75")
    assert first.returncode != 0
    assert "compare-generators freeze" not in " ".join(sandbox.commands())
    sandbox.calls.unlink()
    assert sandbox.run("all", FAKE_STATUS=PASSED).returncode == 0
    assert sandbox.commands()[0] == "gpu=2 compare-generators gate-autoeval"


def test_the_freeze_waits_for_the_gate_and_the_dev_judge(sandbox: ChainSandbox) -> None:
    for step in ("score-train", "score-dev", "pairs", "beta-grid", "train", "select"):
        assert sandbox.run(step).returncode == 0
    result = sandbox.run("freeze")
    assert result.returncode == 2
    assert "step 7 (gate-autoeval) has not completed" in result.stderr


def test_the_gate_cannot_run_after_the_freeze(sandbox: ChainSandbox) -> None:
    assert sandbox.run("all", FAKE_STATUS=PASSED).returncode == 0
    result = sandbox.run("gate-autoeval")
    assert result.returncode == 2
    assert "cannot run after the freeze" in result.stderr


@pytest.mark.parametrize("change", ["calibration", "commit"])
def test_a_rerun_with_another_calibration_or_commit_is_refused(
    sandbox: ChainSandbox, change: str
) -> None:
    assert sandbox.run("score-train").returncode == 0
    if change == "calibration":
        (sandbox.root / "other").mkdir()
        result = sandbox.run("score-dev", PHYSVERIFY_FT_CALIBRATION=str(sandbox.root / "other"))
        key = "FT_CALIBRATION"
    else:
        state = sandbox.run_dir / "state.env"
        text = state.read_text(encoding="utf-8")
        state.write_text(
            "\n".join(
                "FT_COMMIT=deadbeef" if line.startswith("FT_COMMIT=") else line
                for line in text.splitlines()
            )
            + "\n",
            encoding="utf-8",
        )
        result = sandbox.run("score-dev")
        key = "FT_COMMIT"
    assert result.returncode == 2
    assert f"ran with {key}=" in result.stderr


def test_a_relative_invocation_still_reenters_through_the_slot(sandbox: ChainSandbox) -> None:
    result = subprocess.run(
        [str(sandbox.bash), "run_finetune_chain.sh", "score-train"],
        cwd=sandbox.scripts,
        env=sandbox.env,
        capture_output=True,
        text=True,
        timeout=CALL_TIMEOUT_S,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sandbox.commands() == ["gpu=2 finetune score --split train"]
