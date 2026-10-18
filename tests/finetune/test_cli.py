from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from cli_runner import CliRunner

from physverify.cli.commands.finetune import DEFAULT_RUN
from physverify.config.models import Settings
from physverify.finetune.records import (
    CandidateRecord,
    CandidateScore,
    PairRecord,
    read_records,
    write_record,
    write_records,
)
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.finetune.store import PAIRS_FILE, REPORT_FILE, CampaignStore
from physverify.finetune.trainer import STATE_FILE, checkpoint_name
from physverify.schema.checks import CheckResult, Verdict
from physverify.schema.io import write_jsonl

SUBCOMMANDS = (
    ("prompts", "build"),
    ("generate",),
    ("score",),
    ("pairs",),
    ("train",),
    ("beta-grid",),
    ("select",),
    ("budget",),
    ("status",),
)
Score = Callable[..., CandidateScore]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {"PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}


@pytest.fixture
def store(make_settings: Callable[..., Settings]) -> CampaignStore:
    return CampaignStore.from_settings(make_settings())


@pytest.fixture
def labels(monkeypatch: pytest.MonkeyPatch, fake_labels: frozenset[str]) -> None:
    monkeypatch.setattr(
        "physverify.cli.commands.finetune.sizing_labels", lambda settings: fake_labels
    )


@pytest.mark.parametrize("words", SUBCOMMANDS)
def test_every_subcommand_has_help(cli: CliRunner, words: tuple[str, ...]) -> None:
    result = cli("finetune", *words, "--help")
    assert result.code == 0, result.err
    assert "Usage" in result.out


@pytest.mark.parametrize(
    "argv",
    [("generate", "--split", "train"), ("score", "--split", "train")],
)
def test_gpu_subcommands_exit_2_on_cpu(
    cli: CliRunner, env: dict[str, str], argv: tuple[str, ...]
) -> None:
    result = cli("--device", "cpu", "finetune", *argv, env=env)
    assert result.code == 2
    assert "--device cuda" in result.err


def test_prompts_build_writes_and_checks_the_set(
    cli: CliRunner, env: dict[str, str], store: CampaignStore, labels: None
) -> None:
    missing = cli("finetune", "prompts", "build", "--check", env=env)
    assert missing.code == 1
    built = cli("finetune", "prompts", "build", env=env)
    assert built.code == 0, built.err
    assert "sha256" in built.out and "train: 2" in built.out
    assert sum(p.split == "train" for p in store.prompts()) == 2
    assert cli("finetune", "prompts", "build", "--check", env=env).code == 0
    store.prompts_path.write_text("{}\n", encoding="utf-8")
    assert cli("finetune", "prompts", "build", "--check", env=env).code == 1


def test_an_unsized_noun_fails_the_build(
    cli: CliRunner, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("physverify.cli.commands.finetune.sizing_labels", lambda s: {"ball"})
    result = cli("finetune", "prompts", "build", env=env)
    assert result.code == 2
    assert "size prior" in result.err


def _pair(calibrated: bool) -> PairRecord:
    return PairRecord(
        scorer="provisional",
        scorer_version="1",
        calibrated=calibrated,
        thresholds="t",
        pair_id="p__k00__vs__p__k01",
        prompt_id="p",
        winner="p__k00",
        loser="p__k01",
        score_w=0.9,
        score_l=0.5,
        margin=0.4,
        weight=1.0,
        common_checks=["R1"],
    )


def test_train_refuses_provisional_pairs_without_the_flag(
    cli: CliRunner, env: dict[str, str], store: CampaignStore
) -> None:
    write_records(store.pairs_dir("provisional") / PAIRS_FILE, [_pair(False)])
    train = ("finetune", "train", "--scorer", "provisional")
    refused = cli(*train, env=env)
    assert refused.code == 2
    assert "--allow-provisional" in refused.err
    allowed = cli("--device", "cpu", *train, "--allow-provisional", env=env)
    assert allowed.code == 2
    assert "--device cuda" in allowed.err


def test_train_refuses_a_schedule_with_too_few_checkpoints(
    cli: CliRunner, env: dict[str, str], store: CampaignStore
) -> None:
    write_records(store.pairs_dir("provisional") / PAIRS_FILE, [_pair(False)])
    result = cli(
        "--set",
        "finetune.select.checkpoints=5",
        "finetune",
        "train",
        "--scorer",
        "provisional",
        "--allow-provisional",
        env=env,
    )
    assert result.code == 2
    assert "fewer than finetune.select.checkpoints = 5" in result.err


def test_no_resume_refuses_existing_checkpoints_without_force(
    cli: CliRunner, env: dict[str, str], store: CampaignStore
) -> None:
    write_records(store.pairs_dir("provisional") / PAIRS_FILE, [_pair(False)])
    step = store.train_dir(DEFAULT_RUN) / checkpoint_name(1)
    step.mkdir(parents=True)
    (step / STATE_FILE).write_bytes(b"")
    train = ("--device", "cpu", "finetune", "train", "--scorer", "provisional")
    refused = cli(*train, "--allow-provisional", "--no-resume", env=env)
    assert refused.code == 2 and "--force" in refused.err
    forced = cli(*train, "--allow-provisional", "--no-resume", "--force", env=env)
    assert forced.code == 2 and "--device cuda" in forced.err


def _write_stored_scores(
    store: CampaignStore,
    make_score: Score,
    make_result: Callable[..., CheckResult],
    thresholds: str,
    version: str,
) -> None:
    for prompt in [p for p in store.prompts() if p.split == "train"]:
        for k, value in enumerate([0.9, 0.2]):
            cid = f"{prompt.prompt_id}__k{k:02d}"
            checks = store.checks_path("base", "provisional", cid)
            checks.parent.mkdir(parents=True, exist_ok=True)
            results = [
                make_result(check, Verdict.PASS, 1.0 - value, 1.0, clip_id=cid)
                for check in ("R1", "R2")
            ]
            write_jsonl(checks, results)
            record = CandidateRecord(
                candidate_id=cid,
                prompt_id=prompt.prompt_id,
                split="train",
                k=k,
                seed=k,
                model_tag="base",
                video_path=str(store.video_path("base", prompt.prompt_id, k)),
                latents_path=None,
                embeds_path=str(store.embeds_path(prompt.prompt_id)),
                config_hash="h",
            )
            write_record(store.meta_path("base", prompt.prompt_id, k), record)
            score = make_score(
                cid,
                value,
                prompt_id=prompt.prompt_id,
                thresholds=thresholds,
                results_path=str(checks),
                scorer_version=version,
            )
            write_record(store.score_path("base", "provisional", cid), score)


def test_pairs_and_status_run_on_cpu_from_stored_scores(
    cli: CliRunner,
    env: dict[str, str],
    store: CampaignStore,
    labels: None,
    make_score: Score,
    make_result: Callable[..., CheckResult],
    make_settings: Callable[..., Settings],
) -> None:
    assert cli("finetune", "prompts", "build", env=env).code == 0
    thresholds = ProvisionalScorer(make_settings()).thresholds()
    _write_stored_scores(store, make_score, make_result, thresholds, ProvisionalScorer.version)
    result = cli("finetune", "pairs", "--scorer", "provisional", env=env)
    assert result.code == 0, result.err
    pairs = read_records(store.pairs_dir("provisional") / PAIRS_FILE, PairRecord)
    assert len(pairs) == 2 and all(p.winner.endswith("__k00") for p in pairs)
    assert (store.pairs_dir("provisional") / REPORT_FILE).is_file()
    status = cli("finetune", "status", env=env)
    assert status.code == 0
    assert "pairs/provisional: 2" in status.out
    assert "base: candidates" in status.out


def test_pairs_refuse_scores_of_an_older_provisional_version(
    cli: CliRunner,
    env: dict[str, str],
    store: CampaignStore,
    labels: None,
    make_score: Score,
    make_result: Callable[..., CheckResult],
    make_settings: Callable[..., Settings],
) -> None:
    assert cli("finetune", "prompts", "build", env=env).code == 0
    thresholds = ProvisionalScorer(make_settings()).thresholds()
    _write_stored_scores(store, make_score, make_result, thresholds, "1")
    result = cli("finetune", "pairs", "--scorer", "provisional", env=env)
    assert result.code == 2
    assert "new campaign" in result.err


def test_status_without_prompts(cli: CliRunner, env: dict[str, str]) -> None:
    result = cli("finetune", "status", env=env)
    assert result.code == 0 and "no prompts.jsonl" in result.out


def test_budget_refuses_a_directory_that_is_not_a_timing_run(
    cli: CliRunner, env: dict[str, str], tmp_path: Path
) -> None:
    result = cli("finetune", "budget", "--timing-run", str(tmp_path), "--gpu-hours", "10", env=env)
    assert result.code == 2
    assert "not a timing run" in result.err
