from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from physverify.config.models import Settings
from physverify.schema.clip import ClipManifest
from physverify.timing._base import TimingContext
from physverify.timing.harness import IN_PROGRESS, TimingHarness
from physverify.timing.records import (
    SCHEMA_VERSION,
    CallFailure,
    ResultsFile,
    Sample,
    TargetOutcome,
    TargetResult,
    TargetStatus,
    TimingError,
)
from physverify.timing.report import (
    FAILURES,
    RESULTS,
    SAMPLE_COLUMNS,
    SAMPLES,
    STAT_COLUMNS,
    STATS,
    SUMMARY,
    TOTAL_LABEL,
    CsvSampleSink,
    ResultsWriter,
    load_failures,
    load_results,
    load_samples,
    rebuild,
    render_summary,
    write_results,
)
from physverify.timing.settings import TargetRunSettings
from physverify.timing.stats import stat_row, stat_rows
from physverify.timing.target_services import TimingServices

MAX_IQR_REL = 0.10
TOTAL = ("fake.wall_s", "fake.load_s")
RUN = TargetRunSettings(warmup=1, warmup_scope="clip", repetitions=2)
EXPECTED_STAT_COLUMNS = (
    "target",
    "clip_id",
    "measure",
    "unit",
    "n",
    "median",
    "p25",
    "p75",
    "iqr",
    "iqr_rel",
    "mean",
    "sd",
    "min",
    "max",
    "unstable",
)


def results_file(*, valid: bool) -> ResultsFile:
    return ResultsFile(
        valid_measurement=valid,
        validity_reasons=[] if valid else ["device is 'cpu', not cuda"],
        run={"run_id": "r1", "commit": "abc123", "config_hash": "cafe"},
        environment={
            "device": "cuda" if valid else "cpu",
            "gpus": [{"name": "NVIDIA H200"}] if valid else [],
            "packages": {"torch": "2.8", "diffusers": None},
        },
    )


@pytest.fixture
def ctx(make_settings: Callable[..., Settings], tmp_path: Path) -> TimingContext:
    return TimingContext(make_settings(), "cpu", 1, tmp_path)


@pytest.fixture
def clips(clip_factory: Callable[..., ClipManifest]) -> list[ClipManifest]:
    return [clip_factory("c0"), clip_factory("c1")]


@pytest.fixture
def timed(
    make_services: Callable[..., TimingServices],
    ctx: TimingContext,
    clips: list[ClipManifest],
    tmp_path: Path,
) -> Callable[..., Path]:
    """Run fake targets through the real sink and writer; returns the timing directory."""

    def go(targets: Sequence[type[Any]], *, valid: bool = False) -> Path:
        directory = tmp_path / "timing"
        writer = ResultsWriter(directory, results_file(valid=valid))
        with CsvSampleSink(directory) as sink:
            harness = TimingHarness(sink, on_outcome=writer.record)
            plan = [(cls, RUN) for cls in targets]
            harness.run(plan, lambda cls: cast(Any, cls)(make_services()), lambda: clips, ctx)
        return directory

    return go


def csv_header(path: Path) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(next(csv.reader(handle)))


def ramp(n: int, clip: ClipManifest) -> dict[str, float]:
    return {"fake.wall_s": 1.0 + n, "fake.load_s": 0.5}


def test_results_json_validates_and_tracks_each_target(
    timed: Callable[..., Path], make_fake_target: Callable[..., type[Any]]
) -> None:
    directory = timed([make_fake_target("a", script=ramp), make_fake_target("b", script=ramp)])
    results = ResultsFile.model_validate_json((directory / RESULTS).read_text())
    assert results.schema_version == SCHEMA_VERSION
    assert [(t.name, t.status) for t in results.targets] == [
        ("a", TargetStatus.MEASURED),
        ("b", TargetStatus.MEASURED),
    ]
    assert results.targets[0].n_calls == 6


def test_writer_replaces_a_provisional_target(tmp_path: Path) -> None:
    writer = ResultsWriter(tmp_path, results_file(valid=False))
    first = TargetOutcome("a", False, TargetStatus.PARTIAL, IN_PROGRESS, RUN)
    writer.record(first)
    writer.record(TargetOutcome("b", False, TargetStatus.SKIPPED, "no docker", RUN))
    writer.record(TargetOutcome("a", False, TargetStatus.MEASURED, None, RUN, n_calls=3))
    targets = load_results(tmp_path).targets
    assert [(t.name, t.status) for t in targets] == [
        ("a", TargetStatus.MEASURED),
        ("b", TargetStatus.SKIPPED),
    ]


def test_nan_statistics_survive_a_json_round_trip(tmp_path: Path) -> None:
    row = stat_row("a", "c0", "fake.wall_s", "s", [0.0], MAX_IQR_REL)
    outcome = TargetOutcome("a", False, TargetStatus.MEASURED, None, RUN)
    targets = [TargetResult.from_outcome(outcome, [row])]
    write_results(tmp_path, results_file(valid=False).model_copy(update={"targets": targets}))
    assert "NaN" in (tmp_path / RESULTS).read_text()
    loaded = load_results(tmp_path).targets[0].stats[0]
    assert loaded.n == 1 and math.isnan(loaded.sd) and math.isnan(loaded.iqr_rel)


def test_file_headers_are_exact(
    timed: Callable[..., Path], make_fake_target: Callable[..., type[Any]]
) -> None:
    directory = timed([make_fake_target("a", script=ramp)])
    rebuild(directory, MAX_IQR_REL, TOTAL)
    assert STAT_COLUMNS == EXPECTED_STAT_COLUMNS
    assert csv_header(directory / SAMPLES) == SAMPLE_COLUMNS
    assert csv_header(directory / STATS) == EXPECTED_STAT_COLUMNS
    assert {p.name for p in directory.iterdir()} >= {RESULTS, SAMPLES, STATS, SUMMARY}


def test_samples_are_on_disk_while_the_run_goes_and_after_an_interrupt(
    make_fake_target: Callable[..., type[Any]],
    make_services: Callable[..., TimingServices],
    ctx: TimingContext,
    clips: list[ClipManifest],
    tmp_path: Path,
) -> None:
    seen_during_run: list[int] = []

    def script(n: int, clip: ClipManifest) -> Any:
        seen_during_run.append(len(load_samples(tmp_path)))
        return KeyboardInterrupt() if n == 3 else {"fake.wall_s": 1.0}

    cls = make_fake_target("a", script=script)
    writer = ResultsWriter(tmp_path, results_file(valid=False))
    with pytest.raises(KeyboardInterrupt), CsvSampleSink(tmp_path) as sink:
        TimingHarness(sink, on_outcome=writer.record).run(
            [(cls, RUN)], lambda c: cast(Any, c)(make_services()), lambda: clips, ctx
        )
    assert seen_during_run == [0, 1, 2, 3]
    assert len(load_samples(tmp_path)) == 3
    target = load_results(tmp_path).targets[0]
    assert target.status is TargetStatus.PARTIAL and target.reason is not None
    assert "interrupted" in target.reason
    assert cls.journal.closed == 1


def test_rebuild_reproduces_stats_from_samples(
    timed: Callable[..., Path], make_fake_target: Callable[..., type[Any]]
) -> None:
    directory = timed([make_fake_target("a", script=ramp)])
    first = rebuild(directory, MAX_IQR_REL, TOTAL)
    stats = (directory / STATS).read_bytes()
    (directory / STATS).unlink()
    (directory / SUMMARY).unlink()
    again = rebuild(directory, MAX_IQR_REL, TOTAL)
    assert (directory / STATS).read_bytes() == stats
    assert again == first
    expected = stat_rows(load_samples(directory), MAX_IQR_REL)
    assert again.targets[0].stats == expected
    warm = [s for s in load_samples(directory) if s.warmup]
    assert len(warm) == 2 * len(TOTAL)
    assert all(row.n == 2 for row in expected if row.clip_id != "*")


def test_harness_check_summary_has_the_banner_and_no_total(
    timed: Callable[..., Path], make_fake_target: Callable[..., type[Any]]
) -> None:
    directory = timed([make_fake_target("a", script=ramp)], valid=False)
    rebuild(directory, MAX_IQR_REL, TOTAL)
    summary = (directory / SUMMARY).read_text()
    assert summary.splitlines()[0] == (
        "> HARNESS CHECK ONLY: not a cost measurement (device is 'cpu', not cuda). "
        "Per-clip cost is measured only on the H200."
    )
    assert TOTAL_LABEL not in summary


def test_valid_summary_has_the_total_and_no_banner(
    timed: Callable[..., Path], make_fake_target: Callable[..., type[Any]]
) -> None:
    directory = timed([make_fake_target("a", script=ramp, setup={"fake.setup_s": 2.0})], valid=True)
    rebuild(directory, MAX_IQR_REL, TOTAL)
    summary = (directory / SUMMARY).read_text()
    assert "HARNESS CHECK" not in summary
    assert f"{TOTAL_LABEL}: 4.5 s = fake.wall_s 4 + fake.load_s 0.5" in summary
    assert "NVIDIA H200" in summary and "fake.setup_s" in summary and "unstable" in summary


def test_valid_summary_names_missing_total_measures() -> None:
    summary = render_summary(results_file(valid=True), ["generate.b1.per_video_s"])
    assert f"{TOTAL_LABEL}: not available, missing generate.b1.per_video_s" in summary


def test_rebuild_of_a_killed_run_keeps_its_samples_and_failures(tmp_path: Path) -> None:
    writer = ResultsWriter(tmp_path, results_file(valid=False))
    writer.record(TargetOutcome("a", False, TargetStatus.PARTIAL, IN_PROGRESS, RUN))
    with CsvSampleSink(tmp_path) as sink:
        for rep, value in enumerate((9.0, 1.0, 3.0)):
            sink.add(Sample("a", "c0", rep, rep == 0, "fake.wall_s", "s", value))
        sink.fail(CallFailure("a", "c1", 0, "RuntimeError: boom"))
    with (tmp_path / SAMPLES).open("a", encoding="utf-8") as handle:
        handle.write("a,c1,1,fal")
    results = rebuild(tmp_path, MAX_IQR_REL, TOTAL)
    (target,) = results.targets
    assert (target.status, target.reason) == (TargetStatus.PARTIAL, IN_PROGRESS)
    assert target.failures == [CallFailure("a", "c1", 0, "RuntimeError: boom")]
    assert target.n_calls == 4
    (across,) = [row for row in target.stats if row.clip_id == "*"]
    assert across.median == 2.0


def test_malformed_rows_before_the_last_are_an_error(tmp_path: Path) -> None:
    with CsvSampleSink(tmp_path) as sink:
        sink.add(Sample("a", "c0", 0, False, "fake.wall_s", "s", 1.0))
    text = (tmp_path / SAMPLES).read_text()
    (tmp_path / SAMPLES).write_text(
        text + "a,c0,1,maybe,fake.wall_s,s,1.0\n" + text.splitlines()[1] + "\n"
    )
    with pytest.raises(TimingError, match="malformed row 2"):
        load_samples(tmp_path)
    (tmp_path / SAMPLES).write_text("x,y\n")
    with pytest.raises(TimingError, match="header"):
        load_samples(tmp_path)


def test_failures_file_round_trips(tmp_path: Path) -> None:
    failure = CallFailure("a", "c0", 2, "ValueError: x")
    with CsvSampleSink(tmp_path) as sink:
        sink.fail(failure)
    assert json.loads((tmp_path / FAILURES).read_text()) == {
        "target": "a",
        "clip_id": "c0",
        "repetition": 2,
        "error": "ValueError: x",
    }
    assert load_failures(tmp_path) == [failure]


def test_missing_results_is_a_timing_error(tmp_path: Path) -> None:
    with pytest.raises(TimingError, match="not a timing run"):
        load_results(tmp_path)
