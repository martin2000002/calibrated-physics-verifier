from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets.local_video import QueryEntry
from physverify.demo import pipeline
from physverify.demo.overlay import OVERLAY_FILE, SIDE_BY_SIDE_FILE
from physverify.demo.pipeline import (
    SCENES_FILE,
    DemoDeps,
    ReportDeps,
    rebuild_report,
    report_dir,
    run_demo,
    run_scenes,
)
from physverify.demo.report import PROVISIONAL_BANNER, REPORT_JSON, REPORT_MD, rule_info
from physverify.demo.settings import OverlayStyle
from physverify.finetune.records import BASE_TAG, CandidateRecord, PromptRecord, write_record
from physverify.finetune.scorers.provisional import ProvisionalScorer
from physverify.finetune.scoring import ScoringDeps
from physverify.finetune.seeds import candidate_id
from physverify.finetune.store import CampaignStore

SRC = Path(pipeline.__file__).resolve().parents[1]
Videos = Callable[..., Path]
FakeScoring = Callable[[Settings], ScoringDeps]
TUNED = "lora-0123456789ab"


def _prompt(prompt_id: str = "adhoc-0001") -> PromptRecord:
    return PromptRecord(
        prompt_id=prompt_id,
        text="a ball falls off a table",
        queries=[QueryEntry(phrase="ball", expected_count=1)],
        family="adhoc",
        template_id="fixed",
        target_checks=["R2"],
        split="eval",
        tags=["adhoc"],
    )


def _store(settings: Settings, campaign: str = "demo_adhoc") -> CampaignStore:
    return CampaignStore(
        settings.paths.data_dir / "finetune" / campaign,
        settings.paths.raw_dir / "local_video",
        campaign,
    )


def _deps(settings: Settings, store: CampaignStore, scoring: ScoringDeps) -> DemoDeps:
    scorer = ProvisionalScorer(settings)
    report = ReportDeps(scorer, rule_info(scorer), OverlayStyle(), 10**9, {"commit": "c"})
    return DemoDeps(store, scoring, report, commit="c")


def _files(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file()}


def test_a_given_video_is_scored_into_its_own_campaign(
    tmp_path: Path, settings: Settings, write_video: Videos, fake_scoring: FakeScoring
) -> None:
    store = _store(settings)
    out = tmp_path / "out"
    report = run_demo(
        _prompt(), _deps(settings, store, fake_scoring(settings)), out=out, video=write_video()
    )
    assert report.model_tag.startswith("video-") and len(report.model_tag) == 18
    directory = report_dir(out, report)
    assert {p.name for p in directory.iterdir()} == {REPORT_JSON, REPORT_MD, OVERLAY_FILE}
    assert PROVISIONAL_BANNER in (directory / REPORT_MD).read_text(encoding="utf-8")
    assert [p.prompt_id for p in store.prompts()] == ["adhoc-0001"]
    data = settings.paths.data_dir
    written = _files(data)
    allowed = (store.root, store.video_dir(report.model_tag))
    assert all(any(p.is_relative_to(a) for a in allowed) for p in written), written
    assert set(report.timings) == {"video_s", "score_s"}


def test_rerunning_reuses_the_stored_scores(
    tmp_path: Path, settings: Settings, write_video: Videos, fake_scoring: FakeScoring
) -> None:
    store, video = _store(settings), write_video()
    scoring = fake_scoring(settings)
    deps = _deps(settings, store, scoring)
    first = run_demo(_prompt(), deps, out=tmp_path / "out", video=video)
    second = run_demo(_prompt(), deps, out=tmp_path / "out", video=video)
    assert scoring.runner.clips == [candidate_id("adhoc-0001", 0)]  # type: ignore[attr-defined]
    assert first.clip == second.clip


def test_missing_stored_video_is_a_usage_error(
    tmp_path: Path, settings: Settings, fake_scoring: FakeScoring
) -> None:
    deps = _deps(settings, _store(settings), fake_scoring(settings))
    with pytest.raises(UsageError, match="no complete k = 0 video"):
        run_demo(_prompt(), deps, out=tmp_path / "out")


def test_rebuild_reproduces_the_report(
    tmp_path: Path, settings: Settings, write_video: Videos, fake_scoring: FakeScoring
) -> None:
    deps = _deps(settings, _store(settings), fake_scoring(settings))
    report = run_demo(_prompt(), deps, out=tmp_path / "a", video=write_video())
    rebuilt = rebuild_report(report, deps.report, tmp_path / "b")
    assert rebuilt == report
    assert (report_dir(tmp_path / "b", rebuilt) / OVERLAY_FILE).is_file()


def _stored(store: CampaignStore, prompt: PromptRecord, tag: str, video: Path) -> None:
    target = store.video_path(tag, prompt.prompt_id, 0)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(video.read_bytes())
    record = CandidateRecord(
        candidate_id=candidate_id(prompt.prompt_id, 0),
        prompt_id=prompt.prompt_id,
        split="eval",
        k=0,
        seed=0,
        model_tag=tag,
        video_path=str(target),
        latents_path=None,
        embeds_path=str(store.embeds_path(prompt.prompt_id)),
        config_hash="h",
    )
    write_record(store.meta_path(tag, prompt.prompt_id, 0), record)
    store.write_manifest(tag)


def test_scenes_write_two_reports_per_prompt_and_a_side_by_side(
    tmp_path: Path, settings: Settings, write_video: Videos, fake_scoring: FakeScoring
) -> None:
    store = _store(settings, "wan21_smoke")
    prompts = [_prompt("demo-a"), _prompt("demo-b")]
    for prompt in prompts:
        _stored(store, prompt, BASE_TAG, write_video("base.mp4"))
        _stored(store, prompt, TUNED, write_video("ft.mp4", height=64, width=80, shade=40))
    out = tmp_path / "out"
    pairs = run_scenes(prompts, _deps(settings, store, fake_scoring(settings)), TUNED, out)
    assert [(b.model_tag, f.model_tag) for b, f in pairs] == [(BASE_TAG, TUNED)] * 2
    for base, tuned in pairs:
        for report in (base, tuned):
            names = {p.name for p in report_dir(out, report).iterdir()}
            assert names == {REPORT_JSON, REPORT_MD, OVERLAY_FILE}
        assert (out / base.prompt.prompt_id / SIDE_BY_SIDE_FILE).is_file()
    index = (out / SCENES_FILE).read_text(encoding="utf-8")
    assert "## demo-a" in index and "## demo-b" in index and PROVISIONAL_BANNER in index


def _imports(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def test_the_demo_package_never_imports_the_cli() -> None:
    for path in (SRC / "demo").glob("*.py"):
        assert not any(m.startswith("physverify.cli") for m in _imports(path)), path


def test_the_demo_command_does_not_import_the_finetune_command() -> None:
    modules = _imports(SRC / "cli" / "commands" / "demo.py")
    assert "physverify.cli.commands.finetune" not in modules
