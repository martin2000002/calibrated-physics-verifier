from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from cli_runner import CliRunner

from physverify.agreement.prereg import file_sha256
from physverify.cli.commands.h3 import GATE_LINE, GATE_PASSED
from physverify.core.errors import UsageError
from physverify.evaluation.compare import (
    eval_judges,
    external_status,
    require_bound,
    require_stamp,
    task_values,
)
from physverify.evaluation.prereg import H3Preregistration, chain_shas, load_prereg
from physverify.evaluation.records import (
    GateReport,
    GateTask,
    JudgeOutput,
    JudgeRow,
    Usability,
)
from physverify.evaluation.scoring import manifest
from physverify.evaluation.verdict import NO_VERDICT
from physverify.finetune.records import BASE_TAG, write_record

SUBCOMMANDS = (
    "generate",
    "freeze",
    "score",
    "rescore",
    "judge",
    "gate-autoeval",
    "compare",
    "status",
)
H3_MODULE = Path(__file__).resolve().parents[2] / "src/physverify/cli/commands/h3.py"


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_every_subcommand_has_help(cli: CliRunner, name: str) -> None:
    result = cli("h3", name, "--help")
    assert result.code == 0, result.err
    assert "Usage" in result.out


@pytest.mark.parametrize(
    "argv",
    [("generate", "--arm", "base"), ("score", "--arm", "base"), ("judge", "--split", "dev")],
)
def test_gpu_subcommands_exit_2_on_cpu(
    cli: CliRunner, h3_environ: dict[str, str], argv: tuple[str, ...]
) -> None:
    result = cli("--device", "cpu", "h3", *argv, env=h3_environ)
    assert result.code == 2
    assert "--device cuda" in result.err


def test_the_h3_command_does_not_import_the_finetune_command() -> None:
    tree = ast.parse(H3_MODULE.read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "physverify.cli.commands.finetune" not in imported


def test_status_runs_on_an_empty_campaign(cli: CliRunner, h3_environ: dict[str, str]) -> None:
    result = cli("h3", "status", env=h3_environ)
    assert result.code == 0, result.err
    assert "pre-registration: none" in result.out
    assert "gate: not run" in result.out


@pytest.fixture
def frozen(cli: CliRunner, campaign: Any, h3_environ: dict[str, str]) -> H3Preregistration:
    campaign.build()
    result = cli("h3", "freeze", env=h3_environ)
    assert result.code == 0, result.err
    assert "frozen:" in result.out
    return load_prereg(campaign.layout.prereg)


def _score_eval(campaign: Any, prereg: H3Preregistration) -> str:
    """Scores both arms on eval and writes their manifests, as `h3 score` does on the H200."""
    campaign.write_split("eval", scored=True)
    sha256 = file_sha256(campaign.layout.prereg)
    assert sha256 is not None
    for tag in (BASE_TAG, prereg.adapter_tag):
        record = manifest(campaign.store, tag, campaign.scorer, None, sha256)
        write_record(campaign.layout.scoring(tag, campaign.scorer.name), record)
    return sha256


def test_freeze_records_the_state_and_is_idempotent(
    cli: CliRunner, campaign: Any, h3_environ: dict[str, str], frozen: H3Preregistration
) -> None:
    assert frozen.adapter_tag == campaign.ft_tag
    assert frozen.rule_status == "provisional"
    assert frozen.margins == {"pc": 0.13, "sa": 0.05}
    again = cli("h3", "freeze", env=h3_environ)
    assert again.code == 0, again.err
    assert "unchanged" in again.out


def test_the_first_freeze_is_refused_after_eval_scores_exist(
    cli: CliRunner, campaign: Any, h3_environ: dict[str, str]
) -> None:
    campaign.build(eval_scored=True)
    result = cli("h3", "freeze", env=h3_environ)
    assert result.code == 2
    assert "not blind" in result.err
    assert not campaign.layout.prereg.exists()


def test_compare_writes_the_provisional_report_end_to_end(
    cli: CliRunner, campaign: Any, h3_environ: dict[str, str], frozen: H3Preregistration
) -> None:
    _score_eval(campaign, frozen)
    result = cli("h3", "compare", env=h3_environ)
    assert result.code == 0, result.err
    assert f"verdict: {NO_VERDICT}" in result.out
    report = json.loads(campaign.layout.report_json.read_text(encoding="utf-8"))
    assert report["verdict"]["verdict"] == NO_VERDICT
    assert report["audit"]["blind"] is True
    assert report["audit"]["frozen_after_eval"] is False
    assert report["analysis"]["primary"]["n"] == 8
    text = campaign.layout.report_md.read_text(encoding="utf-8")
    for heading in (
        "## Arms",
        "## Primary and external (fixed sequence)",
        "## Secondary tests (Holm, m = 3)",
        "## Guards",
        "## Sensitivities and strata",
        "## Minimum detectable effect and margins",
        "## Freeze audit",
        "## Circularity",
    ):
        assert heading in text
    assert "code commit at the freeze" in text
    status = cli("h3", "status", env=h3_environ)
    assert status.code == 0, status.err
    assert "provisional" in status.out


def test_compare_refuses_scores_without_a_bound_manifest(
    cli: CliRunner, campaign: Any, h3_environ: dict[str, str], frozen: H3Preregistration
) -> None:
    _score_eval(campaign, frozen)
    campaign.layout.scoring(BASE_TAG, campaign.scorer.name).unlink()
    result = cli("h3", "compare", env=h3_environ)
    assert result.code == 2
    assert "h3 score" in result.err


@pytest.mark.parametrize(
    ("version", "label", "accepted"),
    [("2", "provisional", True), ("2", "calibrated:abc", False), ("1", "provisional", False)],
)
def test_require_stamp_refuses_scores_of_another_version_or_label(
    make_score: Any, version: str, label: str, accepted: bool
) -> None:
    scores = [make_score("e0__k00", 0.5)]
    prereg = H3Preregistration.model_construct(scorer_version=version, rule_label=label)
    if accepted:
        require_stamp(scores, prereg, BASE_TAG)
    else:
        with pytest.raises(UsageError, match="scorer version '2'.*rescore"):
            require_stamp(scores, prereg, BASE_TAG)
    with pytest.raises(UsageError, match="no eval scores"):
        require_stamp([], prereg, BASE_TAG)


def test_freeze_records_the_scorer_version(campaign: Any, frozen: H3Preregistration) -> None:
    assert (frozen.scorer_version, frozen.rule_label) == (
        campaign.scorer.version,
        campaign.scorer.thresholds(),
    )


@pytest.mark.parametrize(
    ("passed", "shown"), [(None, "not run"), (False, "failed"), (True, "passed")]
)
def test_status_shows_the_gate_line_run_h3_reads(
    cli: CliRunner, campaign: Any, h3_environ: dict[str, str], passed: bool | None, shown: str
) -> None:
    campaign.build()
    if passed is not None:
        gate = GateReport(
            tasks={}, clip_ids=[], reference_sha256=None, freeze_sha256=None, passed=passed
        )
        campaign.layout.gate.parent.mkdir(parents=True, exist_ok=True)
        write_record(campaign.layout.gate, gate)
    result = cli("h3", "status", env=h3_environ)
    assert result.code == 0, result.err
    assert f"{GATE_LINE}{shown}" in result.out.splitlines()
    assert (f"{GATE_LINE}{GATE_PASSED}" in result.out.splitlines()) is bool(passed)


def test_require_bound_accepts_only_hashes_of_the_chain() -> None:
    require_bound("a", {"a", "b"}, "x")
    with pytest.raises(UsageError, match="not bound"):
        require_bound(None, {"a"}, "x")


def _output(tag: str, stamp: str, sha256: str | None, value: float) -> JudgeOutput:
    return JudgeOutput(
        split="eval",
        model_tag=tag,
        task="pc",
        stamp=stamp,
        scale=(1.0, 5.0),
        prereg_sha256=sha256,
        rows={"e0__k00": JudgeRow(value=value, reason=None)},
    )


def test_task_values_keep_the_arms_of_one_seed_apart() -> None:
    outputs = {
        f"{BASE_TAG}__pc": _output(BASE_TAG, "s", None, 3.0),
        "lora__pc": _output("lora", "s", None, 4.0),
        f"{BASE_TAG}__sa": None,
    }
    assert task_values(outputs, "pc") == {(BASE_TAG, "e0__k00"): 3.0, ("lora", "e0__k00"): 4.0}
    assert task_values(outputs, "sa") == {}


def test_eval_judges_refuse_an_unbound_or_restamped_output(
    campaign: Any, frozen: H3Preregistration
) -> None:
    sha256 = file_sha256(campaign.layout.prereg)
    assert sha256 is not None
    known = chain_shas(campaign.layout, frozen, sha256)
    path = campaign.layout.judge_output("eval", BASE_TAG, "pc")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_record(path, _output(BASE_TAG, frozen.judge_stamps["pc"], "other", 3.0))
    with pytest.raises(UsageError, match="not bound"):
        eval_judges(campaign.layout, [BASE_TAG], frozen, known)
    write_record(path, _output(BASE_TAG, "another-stamp", sha256, 3.0))
    with pytest.raises(UsageError, match="stamp"):
        eval_judges(campaign.layout, [BASE_TAG], frozen, known)
    write_record(path, _output(BASE_TAG, frozen.judge_stamps["pc"], sha256, 3.0))
    found = eval_judges(campaign.layout, [BASE_TAG], frozen, known)
    assert found[f"{BASE_TAG}__pc"] is not None and found[f"{BASE_TAG}__sa"] is None


def test_external_status_follows_the_gate_and_the_dev_usability(
    campaign: Any, frozen: H3Preregistration
) -> None:
    layout = campaign.layout
    assert external_status(layout, frozen, {}) == "not_run"
    gate = GateReport(
        tasks={}, clip_ids=[], reference_sha256=None, freeze_sha256=None, passed=False
    )
    layout.gate.parent.mkdir(parents=True, exist_ok=True)
    write_record(layout.gate, gate)
    assert external_status(layout, frozen, {}) == "failed_gate"
    write_record(layout.gate, gate.model_copy(update={"passed": True}))
    assert frozen.usability is None
    assert external_status(layout, frozen, {}) == "not_run"


def test_external_status_needs_the_gate_of_the_frozen_judge_stamps(
    campaign: Any, frozen: H3Preregistration
) -> None:
    layout = campaign.layout
    usable = frozen.model_copy(update={"usability": Usability(usable=True)})
    outputs = {"base__pc": _output(BASE_TAG, frozen.judge_stamps["pc"], "s", 3.0)}

    def gate(stamps: dict[str, str]) -> GateReport:
        tasks = {
            task: GateTask(
                task=task,
                stamp=stamp,
                n=10,
                exact=5,
                exact_share=0.5,
                mae=0.5,
                parse_failures=0,
                passed=True,
            )
            for task, stamp in stamps.items()
        }
        return GateReport(
            tasks=tasks, clip_ids=[], reference_sha256=None, freeze_sha256=None, passed=True
        )

    layout.gate.parent.mkdir(parents=True, exist_ok=True)
    write_record(layout.gate, gate(dict(frozen.judge_stamps)))
    assert external_status(layout, usable, outputs) == "ok"
    write_record(layout.gate, gate({task: "another-stamp" for task in frozen.judge_stamps}))
    assert external_status(layout, usable, outputs) == "not_run"


ENFORCE = ("--set", "finetune.stop_rule.enforce=true")


@pytest.mark.parametrize("argv", [("freeze",), ("--device", "cpu", "generate")])
def test_a_fired_stop_rule_blocks_the_eval_arm(
    cli: CliRunner, campaign: Any, h3_environ: dict[str, str], argv: tuple[str, ...]
) -> None:
    campaign.build()
    options, command = (argv[:-1], argv[-1:])
    result = cli(*ENFORCE, *options, "h3", *command, env=h3_environ)
    assert result.code == 2
    assert "pilot stop rule fired" in result.err and "not evaluable" in result.err
    assert not campaign.layout.prereg.exists()
    declared = cli(
        *ENFORCE, *options, "h3", *command, "--declare-deviation", "pilot accepted", env=h3_environ
    )
    verdict = json.loads(
        (campaign.store.pairs_dir(campaign.scorer.name) / "stop_rule.json").read_text("utf-8")
    )
    assert verdict["deviation"] == "pilot accepted" and not verdict["passed"]
    assert "pilot stop rule fired" not in declared.err
    if command == ("freeze",):
        assert declared.code == 0, declared.err
    else:
        assert declared.code == 2 and "--device cuda" in declared.err


def test_a_stop_rule_file_of_other_pairs_is_not_enough(
    cli: CliRunner, campaign: Any, h3_environ: dict[str, str]
) -> None:
    campaign.build()
    path = campaign.store.pairs_dir(campaign.scorer.name) / "stop_rule.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"passed": True, "pairs_sha256": "other"}), "utf-8")
    result = cli(*ENFORCE, "h3", "freeze", env=h3_environ)
    assert result.code == 2 and "pilot stop rule fired" in result.err


def test_the_stop_rule_is_off_in_the_fast_profile(
    cli: CliRunner, campaign: Any, h3_environ: dict[str, str]
) -> None:
    campaign.build()
    result = cli("h3", "freeze", env=h3_environ)
    assert result.code == 0, result.err
    assert "pilot stop rule: FIRED (reported only)" in result.err
