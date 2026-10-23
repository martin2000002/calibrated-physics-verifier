"""``physverify diagnose`` and ``physverify data`` (ADR 0007 §4.3, §6)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest
from cli_runner import CliRunner
from synthetic_clip import CLIP_ID, DATASET

from physverify.app import plugins
from physverify.cli.commands import data, diagnose
from physverify.cli.commands.extract import ConsoleObserver, EventRecorder
from physverify.config.models import Settings
from physverify.core.download import MANIFEST_FILE, FetchedFile, FetchManifest, write_manifest
from physverify.core.errors import MissingInputError, UsageError
from physverify.diagnosis.context import DiagnosisContext
from physverify.diagnosis.real_video import ALL_CLIPS, run_real
from physverify.diagnosis.report import RealVideoReport, build_provenance
from physverify.extraction._base import StageProvider
from physverify.schema.checks import Verdict
from physverify.schema.clip import ClipManifest

from .conftest import CALIB_CLIP, FAKE_CHECK_ID, FAKE_PLAN, SplitFixtureDataset

SET_FAKES = tuple(arg for item in FAKE_PLAN for arg in ("--set", item))
REAL = ("physics_iq", "pisa_real")


def fake_real(dataset: str) -> type[Any]:
    """A downloadable dataset whose fetch writes one 2 MB file and its manifest."""

    class FakeReal:
        name: ClassVar[str] = dataset
        fetched: ClassVar[list[str | None]] = []

        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        def fetch(self, subset: str | None, dest: Path) -> None:
            FakeReal.fetched.append(subset)
            manifest = FetchManifest(
                dataset=dataset,
                source="memory",
                revision="r1",
                license="CC0",
                subset=subset or "local",
                fetched_utc="2026-09-24T12:00:00+00:00",
                files=(FetchedFile(relative="a.mp4", size=2_000_000),),
                clips=("a",),
            )
            write_manifest(dest / MANIFEST_FILE, manifest)

    return FakeReal


@pytest.fixture
def fakes_real() -> dict[str, type[Any]]:
    return {name: fake_real(name) for name in REAL}


@pytest.fixture
def registries(
    monkeypatch: pytest.MonkeyPatch,
    providers: Mapping[str, type[StageProvider]],
    fakes_real: Mapping[str, type[Any]],
) -> None:
    """The real registries with the heavy fakes, the split fixture and fake real datasets."""
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {
        **kinds,
        "providers": lambda: dict(providers),
        "datasets": lambda: {
            **datasets(),
            SplitFixtureDataset.name: SplitFixtureDataset,
            **fakes_real,
        },
    }
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


def run_dir_of(output: str) -> Path:
    (line,) = [line for line in output.splitlines() if line.startswith("run: ")]
    return Path(line.removeprefix("run: "))


def test_help_lists_every_subcommand(cli: CliRunner) -> None:
    result = cli("diagnose", "--help")
    assert result.code == 0
    for name in ("stages", "end-to-end", "attribution", "real"):
        assert name in result.out
    fetch = cli("data", "--help")
    assert fetch.code == 0 and "fetch" in fetch.out


# --- data fetch -------------------------------------------------------------------------------


def test_only_the_real_datasets_can_be_fetched(cli: CliRunner) -> None:
    result = cli("data", "fetch", "kubric_synthetic")
    assert result.code == 2
    assert "cannot be downloaded" in result.err


def test_fetch_targets() -> None:
    registry = {"physics_iq": int, "pisa_real": str, "kubric_synthetic": float}
    assert data.fetch_targets(registry, REAL, None) == [int, str]
    assert data.fetch_targets(registry, REAL, "pisa_real") == [str]
    with pytest.raises(UsageError, match="not registered"):
        data.fetch_targets({"physics_iq": int}, REAL, None)


@pytest.mark.usefixtures("registries")
def test_fetch_summarises_every_dataset(
    cli: CliRunner, fakes_real: Mapping[str, type[Any]]
) -> None:
    result = cli("data", "fetch", "--subset", "local")
    assert result.code == 0, result.err
    lines = result.out.splitlines()
    assert [line.split(" ")[0] for line in lines] == list(REAL)
    assert "[local]: 1 clips, 1 files, 2.0 MB" in lines[0]
    assert all(cls.fetched == ["local"] for cls in fakes_real.values())


def test_fetch_all_reports_the_elapsed_time(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(fakes=False)
    lines: list[str] = []
    ticks = iter([10.0, 12.5])
    data.fetch_all([fake_real("physics_iq")], settings, None, lines.append, lambda: next(ticks))
    dest = settings.paths.raw_dir / "physics_iq"
    assert lines == [f"physics_iq [local]: 1 clips, 1 files, 2.0 MB, 2.5 s -> {dest}"]
    assert data.fetch_summary(settings.paths.raw_dir / "none", 1.0) == (
        "none: no manifest written (1.0 s)"
    )


# --- diagnose ---------------------------------------------------------------------------------


def empty_report(ctx: DiagnosisContext) -> RealVideoReport:
    provenance = build_provenance(
        ctx.run,
        dataset=DATASET,
        dataset_versions={},
        subset=ctx.diag.subset,
        error_profile=None,
        clips=(),
    )
    return RealVideoReport(
        provenance=provenance,
        motion=(),
        falls=(),
        verdicts=(),
        thresholds_provisional=True,
        failures=(),
    )


@pytest.mark.usefixtures("registries")
def test_stages_passes_its_options_and_writes_report_and_events(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], bool, bool]] = []

    def stub(
        ctx: DiagnosisContext,
        clips: Sequence[ClipManifest],
        *,
        write_profile: bool,
        replace_mismatched: bool,
    ) -> RealVideoReport:
        calls.append(([c.clip_id for c in clips], write_profile, replace_mismatched))
        return empty_report(ctx)

    monkeypatch.setattr(diagnose, "run_stage_errors", stub)
    argv = ("diagnose", "stages", "--dataset", DATASET, "--clips", CLIP_ID, "--write-profile")
    result = cli(*SET_FAKES, *argv)
    assert result.code == 0, result.err
    assert calls == [([CLIP_ID], True, False)]
    assert cli(*SET_FAKES, *argv, "--replace-mismatched").code == 0
    assert calls[-1] == ([CLIP_ID], True, True)
    run_dir = run_dir_of(result.out)
    assert (run_dir / diagnose.EVENTS_FILE).is_file()
    assert (run_dir / "diagnosis" / "stages.json").is_file()
    assert f"report: {run_dir / 'diagnosis' / 'stages.md'}" in result.out


@pytest.mark.usefixtures("registries")
def test_real_on_the_fixture_reports_falls(cli: CliRunner) -> None:
    result = cli(*SET_FAKES, "diagnose", "real", "--dataset", DATASET, "--clips", CLIP_ID)
    assert result.code == 0, result.err
    report = json.loads((run_dir_of(result.out) / "diagnosis" / "real.json").read_text())
    assert [row["clip_id"] for row in report["falls"]] == [CLIP_ID]
    assert f"{CLIP_ID}: g [estimated]" in result.out


@pytest.mark.usefixtures("registries")
@pytest.mark.parametrize(
    "argv",
    [
        ("diagnose", "stages", "--dataset", "nope"),
        ("diagnose", "stages", "--dataset", DATASET, "--clips", "nope"),
        ("diagnose", "real", "--dataset", DATASET, "--clips", "nope"),
    ],
)
def test_unknown_dataset_or_clip_is_a_usage_error(cli: CliRunner, argv: tuple[str, ...]) -> None:
    result = cli(*SET_FAKES, *argv)
    assert result.code == 2, result.err


def test_real_expands_all_to_the_configured_datasets(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
) -> None:
    ctx = make_diagnosis_context(make_settings())
    assert diagnose.real_datasets(ctx, diagnose.ALL_DATASETS) == REAL
    assert diagnose.real_datasets(ctx, None) == REAL
    assert diagnose.real_datasets(ctx, "pisa_real") == ("pisa_real",)


def test_run_real_counts_verdicts_and_summarises(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
    fake_suite: Callable[..., Any],
) -> None:
    suite = fake_suite(thresholds={FAKE_CHECK_ID: 1e9})
    ctx = make_diagnosis_context(make_settings(), suite=suite)
    report = run_real(ctx, (DATASET,), "local")
    assert [r.clip_id for r in report.falls] == sorted([CALIB_CLIP, CLIP_ID])
    (verdicts,) = report.verdicts
    assert verdicts.check_id == FAKE_CHECK_ID
    assert verdicts.counts[Verdict.PASS] == sum(verdicts.counts.values()) > 0
    assert verdicts.fail_rate == 0.0
    assert report.thresholds_provisional
    assert "fail_rate" in report.provenance.provisional
    assert "g_est" in report.provenance.estimated
    lines = diagnose.summary_lines(report)
    assert any(line.startswith(f"{DATASET} {FAKE_CHECK_ID}: fail rate") for line in lines)


def test_run_real_without_a7_records_the_missing_suite(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
) -> None:
    report = run_real(make_diagnosis_context(make_settings()), (DATASET,), "local", (CLIP_ID,))
    assert report.verdicts == ()
    assert [f.clip_id for f in report.failures] == [ALL_CLIPS]
    assert isinstance(report, RealVideoReport)
    assert diagnose.summary_lines(report)[0].startswith(f"FAILED {ALL_CLIPS}")


def test_an_aborted_level_still_writes_its_events(
    make_settings: Callable[..., Settings],
    make_diagnosis_context: Callable[..., DiagnosisContext],
) -> None:
    ctx = make_diagnosis_context(make_settings())
    recorder = EventRecorder(ConsoleObserver(lambda _: None))

    def explode(_: DiagnosisContext) -> RealVideoReport:
        raise MissingInputError("boom")

    with pytest.raises(MissingInputError, match="boom"):
        diagnose.execute(ctx, recorder, diagnose.REAL, explode)
    assert (ctx.run.run_dir / diagnose.EVENTS_FILE).is_file()
    assert not (ctx.run.run_dir / "diagnosis" / "real.json").exists()


@pytest.mark.usefixtures("registries")
def test_a_missing_input_exits_1_and_still_writes_the_events(cli: CliRunner) -> None:
    argv = ("diagnose", "end-to-end", "--dataset", DATASET, "--clips", CALIB_CLIP)
    result = cli(*SET_FAKES, *argv)
    assert result.code == 1, result.err
    assert "none of the 1 selected clips is in split eval" in result.err
    run_dir = run_dir_of(result.out)
    assert (run_dir / diagnose.EVENTS_FILE).read_text(encoding="utf-8") == ""
    assert not (run_dir / "diagnosis" / "end_to_end.json").exists()
