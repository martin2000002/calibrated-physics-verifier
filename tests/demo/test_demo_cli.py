from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from cli_runner import CliRunner

from physverify.config.models import Settings
from physverify.datasets.local_video import QueryEntry
from physverify.demo.overlay import OVERLAY_FILE
from physverify.demo.report import PROVISIONAL_BANNER, REPORT_JSON, REPORT_MD, read_report
from physverify.finetune.candidate_scoring import ScoringDeps
from physverify.finetune.records import PromptRecord, read_records
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.generator_comparison.prereg import H3Preregistration

SERVICES = "physverify.app.demo"
ADHOC = ("demo", "run", "--prompt", "A ball falls off a table", "--query", "ball:1")
FakeScoring = Callable[[Settings], ScoringDeps]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {"PHYSVERIFY__PATHS__DATA_DIR": str(tmp_path / "data")}


@pytest.fixture
def fakes(
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
    fake_scoring: FakeScoring,
) -> None:
    monkeypatch.setattr(f"{SERVICES}.sizing_labels", lambda settings: frozenset({"ball"}))
    monkeypatch.setattr(f"{SERVICES}.scoring_deps", lambda context: fake_scoring(make_settings()))


@pytest.mark.parametrize("command", ["run", "scenes", "report"])
def test_every_subcommand_has_help(cli: CliRunner, command: str) -> None:
    result = cli("demo", command, "--help")
    assert result.code == 0, result.err
    assert "Usage" in result.out


def test_generation_needs_cuda(cli: CliRunner, env: dict[str, str]) -> None:
    result = cli("--device", "cpu", *ADHOC, env=env)
    assert result.code == 2
    assert "--device cuda" in result.err


def test_a_given_video_runs_on_cpu_into_the_adhoc_campaign(
    cli: CliRunner,
    env: dict[str, str],
    tmp_path: Path,
    write_video: Callable[..., Path],
    fakes: None,
) -> None:
    video = write_video()
    result = cli("--profile", "fast", "--device", "cpu", *ADHOC, "--video", str(video), env=env)
    assert result.code == 0, result.err
    data = tmp_path / "data"
    written = {p for p in data.rglob("*") if p.is_file()}
    roots = (data / "finetune" / "demo_adhoc", data / "raw" / "local_video", data / "demo")
    assert all(any(p.is_relative_to(r) for r in roots) for p in written), written
    assert all(
        p.parent.name.startswith("demo_adhoc__")
        for p in written
        if p.is_relative_to(data / "raw" / "local_video")
    )
    (record,) = read_records(data / "finetune" / "demo_adhoc" / "prompts.jsonl", PromptRecord)
    assert record.prompt_id.startswith("adhoc-") and len(record.prompt_id) == 14
    assert (record.split, record.tags, record.family) == ("eval", ["adhoc"], "adhoc")
    (query,) = record.queries
    assert isinstance(query, QueryEntry) and (query.phrase, query.expected_count) == ("ball", 1)
    assert record.target_checks
    (directory,) = (data / "demo").iterdir()
    assert {p.name for p in directory.iterdir()} == {REPORT_JSON, REPORT_MD, OVERLAY_FILE}
    report = (directory / REPORT_MD).read_text(encoding="utf-8")
    assert PROVISIONAL_BANNER in report and "coverage" in report


def test_unknown_checks_and_missing_queries_are_usage_errors(
    cli: CliRunner, env: dict[str, str], write_video: Callable[..., Path], fakes: None
) -> None:
    video = str(write_video())
    bad = cli("--device", "cpu", *ADHOC, "--target-check", "R99", "--video", video, env=env)
    assert bad.code == 2 and "R99" in bad.err
    bare = ("demo", "run", "--prompt", "p", "--video", video)
    assert cli("--device", "cpu", *bare, env=env).code == 2


def test_adapter_and_video_together_are_a_usage_error(
    cli: CliRunner, env: dict[str, str], tmp_path: Path, write_video: Callable[..., Path]
) -> None:
    video = str(write_video())
    result = cli("--device", "cpu", *ADHOC, "--video", video, "--adapter", str(tmp_path), env=env)
    assert result.code == 2 and "--adapter" in result.err
    assert not (tmp_path / "data" / "demo").exists()


def test_a_query_without_a_size_prior_is_refused(
    cli: CliRunner,
    env: dict[str, str],
    tmp_path: Path,
    write_video: Callable[..., Path],
    fakes: None,
) -> None:
    video = str(write_video())
    args = ("demo", "run", "--prompt", "p", "--query", "zorblax:1", "--video", video)
    result = cli("--device", "cpu", *args, env=env)
    assert result.code == 2 and "zorblax" in result.err
    assert not (tmp_path / "data" / "demo").exists()


def test_report_rebuilds_all_three_files(
    cli: CliRunner,
    env: dict[str, str],
    tmp_path: Path,
    write_video: Callable[..., Path],
    fakes: None,
) -> None:
    video = str(write_video())
    assert cli("--device", "cpu", *ADHOC, "--video", video, env=env).code == 0
    (directory,) = (tmp_path / "data" / "demo").iterdir()
    out = tmp_path / "rebuilt"
    result = cli("demo", "report", "--results", str(directory), "--out", str(out), env=env)
    assert result.code == 0, result.err
    rebuilt = out / directory.name
    assert {p.name for p in rebuilt.iterdir()} == {REPORT_JSON, REPORT_MD, OVERLAY_FILE}
    assert read_report(rebuilt).clip == read_report(directory).clip


def test_scenes_wait_for_the_preregistration(
    cli: CliRunner, env: dict[str, str], tmp_path: Path
) -> None:
    result = cli("demo", "scenes", "--from-select", env=env)
    assert result.code == 2
    assert "h3 freeze" in result.err
    assert not (tmp_path / "data" / "demo").exists()


def _scene_campaign(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "finetune" / "wan21_v1"
    (root / "h3").mkdir(parents=True)
    (root / "h3" / "preregistration.json").write_text("{}", encoding="utf-8")
    prompt = PromptRecord(
        prompt_id="demo-x",
        text="t",
        queries=["ball"],
        family="fall",
        template_id="fixed",
        target_checks=["R2"],
        split="eval",
        tags=["demo"],
    )
    (root / "prompts.jsonl").write_text(prompt.model_dump_json() + "\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "w.bin").write_bytes(b"w")
    return adapter


def test_scenes_need_the_generated_videos(
    cli: CliRunner, env: dict[str, str], tmp_path: Path, fakes: None
) -> None:
    adapter = _scene_campaign(tmp_path)
    result = cli("--profile", "precise", "demo", "scenes", "--adapter", str(adapter), env=env)
    assert result.code == 2
    assert "h3 generate" in result.err and "demo-x/base" in result.err
    assert cli("demo", "scenes", env=env).code == 2


@pytest.mark.parametrize(
    ("available", "frozen_version", "frozen_label"),
    [(True, "2", "frozen-label"), (True, "0", None), (False, "2", "frozen-label")],
    ids=["label-mismatch", "version-mismatch", "unavailable"],
)
def test_scenes_never_fall_back_from_the_frozen_rule(
    cli: CliRunner,
    env: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_settings: Callable[..., Settings],
    fakes: None,
    available: bool,
    frozen_version: str,
    frozen_label: str | None,
) -> None:
    adapter = _scene_campaign(tmp_path)
    label = frozen_label or ProvisionalScorer(make_settings(profile="precise"), {}).thresholds()
    frozen = H3Preregistration.model_construct(
        rule="calibrated", scorer_version=frozen_version, rule_label=label
    )
    monkeypatch.setattr(f"{SERVICES}.require_videos", lambda store, prompts, tags: None)
    monkeypatch.setattr(f"{SERVICES}.load_prereg", lambda path: frozen)
    if available:
        monkeypatch.setattr(
            f"{SERVICES}.scorer_for",
            lambda context, name: ProvisionalScorer(context.settings, {}),
        )
    before = sorted((tmp_path / "data").rglob("*"))
    result = cli("--profile", "precise", "demo", "scenes", "--adapter", str(adapter), env=env)
    assert result.code == 2
    assert ("refreeze" if available else "unavailable") in result.err
    assert sorted((tmp_path / "data").rglob("*")) == before
