from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from cli_runner import CliRunner
from wiring import build_scorer

from physverify.cli.commands.finetune import (
    DEFAULT_RUN,
    _beta,
    _check_current,
    _check_resume,
    status_lines,
)
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.finetune.layout import PAIRS_FILE, REPORT_FILE, CampaignStore
from physverify.finetune.records import (
    CandidateRecord,
    CandidateScore,
    PairRecord,
    PairsReport,
    read_records,
    write_record,
    write_records,
)
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.finetune.settings import load_finetune
from physverify.finetune.trainer import STATE_FILE, checkpoint_name
from physverify.runtime.runs import RUN_FILE
from physverify.schema.checks import CheckResult, Verdict
from physverify.schema.io import write_jsonl

SUBCOMMANDS = (
    ("prompts", "build"),
    ("generate",),
    ("score",),
    ("pairs",),
    ("train",),
    ("beta-grid",),
    ("pilot-review",),
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


@pytest.mark.parametrize("step", ["generate", "score"])
def test_the_eval_split_is_left_to_h3(
    cli: CliRunner, env: dict[str, str], tmp_path: Path, step: str
) -> None:
    result = cli("--device", "cuda", "finetune", step, "--split", "eval", env=env)
    assert result.code == 2
    assert f"physverify h3 {step}" in result.err
    assert not (tmp_path / "data").exists() or not any((tmp_path / "data").rglob("*.json*"))


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
    candidate_hash: str = "h",
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
                candidate_config_hash=candidate_hash,
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
    thresholds = build_scorer("provisional", make_settings()).thresholds()
    _write_stored_scores(store, make_score, make_result, thresholds, ProvisionalScorer.version)
    result = cli("finetune", "pairs", "--scorer", "provisional", env=env)
    assert result.code == 0, result.err
    pairs = read_records(store.pairs_dir("provisional") / PAIRS_FILE, PairRecord)
    assert len(pairs) == 2 and all(p.winner.endswith("__k00") for p in pairs)
    assert (store.pairs_dir("provisional") / REPORT_FILE).is_file()
    assert "1.000 per scored prompt (2 scored)" in result.out
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
    thresholds = build_scorer("provisional", make_settings()).thresholds()
    _write_stored_scores(store, make_score, make_result, thresholds, "1")
    result = cli("finetune", "pairs", "--scorer", "provisional", env=env)
    assert result.code == 2
    assert "finetune score --split train" in result.err


def test_pairs_refuse_scores_of_another_candidate_config(
    cli: CliRunner,
    env: dict[str, str],
    store: CampaignStore,
    labels: None,
    make_score: Score,
    make_result: Callable[..., CheckResult],
    make_settings: Callable[..., Settings],
) -> None:
    assert cli("finetune", "prompts", "build", env=env).code == 0
    thresholds = build_scorer("provisional", make_settings()).thresholds()
    version = ProvisionalScorer.version
    _write_stored_scores(store, make_score, make_result, thresholds, version, "old")
    result = cli("finetune", "pairs", "--scorer", "provisional", env=env)
    assert result.code == 2 and "stale" in result.err


def test_pairs_ignore_stale_scores_of_prompts_outside_train(
    cli: CliRunner,
    env: dict[str, str],
    store: CampaignStore,
    labels: None,
    make_score: Score,
    make_result: Callable[..., CheckResult],
    make_settings: Callable[..., Settings],
) -> None:
    assert cli("finetune", "prompts", "build", env=env).code == 0
    thresholds = build_scorer("provisional", make_settings()).thresholds()
    _write_stored_scores(store, make_score, make_result, thresholds, ProvisionalScorer.version)
    restamped = make_score("eval-0001__k00", 0.5, scorer_version="frozen", thresholds="frozen")
    write_record(store.score_path("base", "provisional", restamped.candidate_id), restamped)
    result = cli("finetune", "pairs", "--scorer", "provisional", env=env)
    assert result.code == 0, result.err


def test_the_pilot_draws_train_prompts_only(cli: CliRunner, env: dict[str, str]) -> None:
    result = cli(
        "--device", "cuda", "finetune", "generate", "--split", "dev", "--pilot", "2", env=env
    )
    assert result.code == 2 and "--pilot" in result.err


def test_select_refuses_stale_checkpoint_scores(
    store: CampaignStore, make_score: Score, make_settings: Callable[..., Settings]
) -> None:
    scorer = ProvisionalScorer(make_settings(), {})
    tag = "lora-abc"
    record = CandidateRecord(
        candidate_id="d__k00",
        prompt_id="d",
        split="dev",
        k=0,
        seed=0,
        model_tag=tag,
        video_path="v.mp4",
        latents_path=None,
        embeds_path="e",
        config_hash="h",
    )
    write_record(store.meta_path(tag, "d", 0), record)
    fields = {"model_tag": tag, "thresholds": scorer.thresholds(), "candidate_config_hash": "h"}
    current = make_score("d__k00", 0.5, scorer_version=scorer.version, **fields)
    _check_current(store, tag, [current], scorer, "dev")
    old = current.model_copy(update={"scorer_version": "0"})
    with pytest.raises(UsageError, match=f"--split dev --model-tag {tag}"):
        _check_current(store, tag, [old], scorer, "dev")


def test_resume_compares_every_run_field_but_commit_and_beta_source(tmp_path: Path) -> None:
    path = tmp_path / RUN_FILE
    record = {
        "pairs_sha256": "p",
        "train": {"lr": 1e-4, "seed": 0},
        "lora": {"rank": 16},
        "base_model": {"revision": "r"},
        "commit": "c1",
        "beta_source": "--beta",
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    _check_resume(path, {**record, "commit": "c2", "beta_source": "beta_grid.json"})
    changes = [
        ("train", {"lr": 5e-5, "seed": 0}),
        ("lora", {"rank": 8}),
        ("base_model", {"revision": "s"}),
        ("pairs_sha256", "q"),
    ]
    for key, value in changes:
        with pytest.raises(UsageError, match=key):
            _check_resume(path, {**record, key: value})
    with pytest.raises(UsageError, match="new_key"):
        _check_resume(path, {**record, "new_key": 1})


def test_a_run_from_before_base_model_tracking_must_be_restarted(tmp_path: Path) -> None:
    path = tmp_path / RUN_FILE
    record = {"pairs_sha256": "p", "train": {"lr": 1e-4}, "commit": "c1"}
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(UsageError, match="predates base-model tracking.*must be restarted"):
        _check_resume(path, {**record, "base_model": {"revision": "r"}})


def test_status_reads_a_pairs_report_without_the_scored_prompt_yield(
    store: CampaignStore, cli: CliRunner, env: dict[str, str]
) -> None:
    assert cli("finetune", "prompts", "build", env=env).code == 0
    report = PairsReport(
        scorer="provisional",
        scorer_version="1",
        calibrated=False,
        thresholds="t",
        prompts=4,
        candidates=16,
        eligible=12,
        pairs=2,
        dropped={},
        rejected={},
        yield_per_prompt=0.5,
    )
    path = store.pairs_dir("provisional") / REPORT_FILE
    path.parent.mkdir(parents=True)
    path.write_text(report.model_dump_json(exclude={"scored_prompts", "yield_per_scored_prompt"}))
    loaded = PairsReport.model_validate_json(path.read_text())
    assert loaded.measured_yield == 0.5
    lines = status_lines(store)
    assert any("pairs/provisional: 2 (0.500 per prompt (4)" in line for line in lines), lines
    assert any("older report" in line for line in lines)


def test_the_beta_grid_choice_is_tied_to_its_pairs(
    store: CampaignStore, make_settings: Callable[..., Settings]
) -> None:
    cfg = load_finetune(make_settings())
    assert _beta(store, cfg, None, "p") == (cfg.train.beta, "finetune.train.beta")
    store.beta_path().parent.mkdir(parents=True)
    store.beta_path().write_text(json.dumps({"chosen": 500.0, "pairs_sha256": "p"}), "utf-8")
    assert _beta(store, cfg, None, "p") == (500.0, str(store.beta_path()))
    assert _beta(store, cfg, 7.0, "q") == (7.0, "--beta")
    with pytest.raises(UsageError, match="other pairs"):
        _beta(store, cfg, None, "q")


def test_status_without_prompts(cli: CliRunner, env: dict[str, str]) -> None:
    result = cli("finetune", "status", env=env)
    assert result.code == 0 and "no prompts.jsonl" in result.out


def test_budget_refuses_a_directory_that_is_not_a_timing_run(
    cli: CliRunner, env: dict[str, str], tmp_path: Path
) -> None:
    result = cli("finetune", "budget", "--timing-run", str(tmp_path), "--gpu-hours", "10", env=env)
    assert result.code == 2
    assert "not a timing run" in result.err


def _stop_pairs(store: CampaignStore, thresholds: str, n_pairs: int, scored: int) -> None:
    """``n_pairs`` stored pairs of the active provisional stamp out of ``scored`` prompts."""
    stamp = {"scorer": "provisional", "scorer_version": ProvisionalScorer.version}
    stamp |= {"calibrated": False, "thresholds": thresholds}
    pairs = [
        _pair(False).model_copy(update={**stamp, "pair_id": f"p{i:02d}", "prompt_id": f"p{i:02d}"})
        for i in range(n_pairs)
    ]
    write_records(store.pairs_dir("provisional") / PAIRS_FILE, pairs)
    report = PairsReport(
        **stamp,
        prompts=scored,
        scored_prompts=scored,
        candidates=4 * scored,
        eligible=4 * scored,
        pairs=n_pairs,
        dropped={},
        rejected={},
        yield_per_prompt=n_pairs / scored,
        yield_per_scored_prompt=n_pairs / scored,
    )
    write_record(store.pairs_dir("provisional") / REPORT_FILE, report)


@pytest.mark.parametrize(("n_pairs", "fires"), [(10, True), (12, False)])
def test_train_obeys_the_pilot_stop_rule(
    cli: CliRunner,
    env: dict[str, str],
    store: CampaignStore,
    labels: None,
    make_score: Score,
    make_result: Callable[..., CheckResult],
    make_settings: Callable[..., Settings],
    n_pairs: int,
    fires: bool,
) -> None:
    assert cli("finetune", "prompts", "build", env=env).code == 0
    thresholds = build_scorer("provisional", make_settings()).thresholds()
    _write_stored_scores(store, make_score, make_result, thresholds, ProvisionalScorer.version)
    _stop_pairs(store, thresholds, n_pairs, 40)
    review = ("finetune", "pilot-review", "--scorer", "provisional")
    sample = cli(*review, "--sample", env=env)
    assert sample.code == 0 and len(sample.out.splitlines()) == 10
    assert cli(*review, "--better", "6", env=env).code == 0
    enforce = ("--set", "finetune.stop_rule.enforce=true", "--device", "cpu")
    train = (*enforce, "finetune", "train", "--scorer", "provisional", "--allow-provisional")
    result = cli(*train, env=env)
    verdict = json.loads((store.pairs_dir("provisional") / "stop_rule.json").read_text())
    assert result.code == 2
    if fires:
        assert "pilot stop rule fired: pairs per scored prompt 0.25 < 0.3" in result.err
        assert not verdict["passed"] and verdict["deviation"] is None
        declared = cli(*train, "--stop-rule-deviation", "pilot too small", env=env)
        assert declared.code == 2 and "--device cuda" in declared.err
        kept = json.loads((store.pairs_dir("provisional") / "stop_rule.json").read_text())
        assert kept["deviation"] == "pilot too small"
    else:
        assert "--device cuda" in result.err and "stop rule" not in result.err
        assert verdict["passed"] and verdict["measured"]["better"] == 6


def test_the_stop_rule_is_only_reported_when_not_enforced(
    cli: CliRunner, env: dict[str, str], store: CampaignStore
) -> None:
    write_records(store.pairs_dir("provisional") / PAIRS_FILE, [_pair(False)])
    train = ("--device", "cpu", "finetune", "train", "--scorer", "provisional")
    result = cli(*train, "--allow-provisional", env=env)
    assert result.code == 2 and "--device cuda" in result.err
    assert "pilot stop rule: FIRED (reported only)" in result.err
    verdict = json.loads((store.pairs_dir("provisional") / "stop_rule.json").read_text())
    assert not verdict["passed"] and not verdict["enforced"]


def test_pilot_review_needs_one_mode(cli: CliRunner, env: dict[str, str]) -> None:
    result = cli("finetune", "pilot-review", "--scorer", "provisional", env=env)
    assert result.code == 2 and "--sample or --better" in result.err
