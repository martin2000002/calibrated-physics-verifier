"""``calibrate dev-score`` and ``calibrate fit`` over a dataset that declares a calibration
extension (a ``checks_only`` subset): its clips join the calib run, its scenes count for R5b only,
and the other checks come out as without it."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest
from calibration_cases import CHECKS, DATASET, extension_report, merged, missing, synthetic_report
from cli_runner import CliRunner
from verify_runs import write_verify_run

from physverify.app import calibration as app_calibration
from physverify.app import plugins
from physverify.config.models import Settings
from physverify.datasets._base import GroundTruthStore, ScopedSubset, VideoSource
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.verification.runner import VerificationReport

BASE = synthetic_report(24)
EXTRA = extension_report()
ALL = merged(BASE, EXTRA)
EXTENSION = "calib_extra"
OTHERS = ("R1", "R3a")
FAST = ("--set", "calibration.bootstrap=30", "--set", "calibration.folds=3")


class PlainDataset:
    """The base calib clips under every subset; ``clips(None)`` also lists the extension's."""

    name: ClassVar[str] = DATASET

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return ALL.clips if subset is None else BASE.clips

    def video(self, clip: ClipManifest) -> VideoSource:
        raise NotImplementedError

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


class ExtendedDataset(PlainDataset):
    def scoped_subsets(self) -> tuple[ScopedSubset, ...]:
        return (ScopedSubset(EXTENSION, ("R5b",), EXTRA.clips),)


class _SubsetRunner:
    """The rows and results of the clips it is given."""

    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    def run(self, source: object, clips: Sequence[ClipManifest]) -> VerificationReport:
        self.calls.append(len(clips))
        return only(ALL, {c.clip_id for c in clips})


class _UncachedRunner(_SubsetRunner):
    """The extension's clips come back as the verify runner returns a clip whose heavy stage is
    not in the cache: every check abstains MISSING_INPUT and the run goes on."""

    def run(self, source: object, clips: Sequence[ClipManifest]) -> VerificationReport:
        found = super().run(source, clips)
        lost = {c.clip_id for c in EXTRA.clips}
        kept = only(found, {c.clip_id for c in clips} - lost)
        absent = tuple(
            replace(missing(u.check_id, c.clip_id, InputLevel.ORACLE), details={"error": UNCACHED})
            for c in clips
            if c.clip_id in lost
            for u in CHECKS
        )
        return replace(kept, results=kept.results + absent, clips=found.clips)


UNCACHED = "a heavy stage is not cached; run `physverify extract run --dataset x` first"


def only(report: VerificationReport, ids: set[str]) -> VerificationReport:
    return VerificationReport(
        results=tuple(r for r in report.results if r.clip_id in ids),
        table=tuple(r for r in report.table if r.clip_id in ids),
        clips=tuple(c for c in report.clips if c.clip_id in ids),
    )


def use(monkeypatch: pytest.MonkeyPatch, dataset: type[PlainDataset]) -> None:
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {**kinds, "datasets": lambda: {**datasets(), DATASET: dataset}}
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    found: list[int] = []
    monkeypatch.setattr(app_calibration, "_raw_runner", lambda session: _SubsetRunner(found))
    monkeypatch.setattr(app_calibration, "_input_source", lambda session, dataset, level: None)
    return found


def devscore(cli: CliRunner, *extra: str) -> tuple[str, dict[str, Any]]:
    result = cli(*FAST, "calibrate", "dev-score", "--dataset", DATASET, "--level", "oracle", *extra)
    assert result.code == 0, result.err
    run_dir = Path(result.out.split("run: ")[1].split()[0])
    return result.out, json.loads((run_dir / "devscore.json").read_text(encoding="utf-8"))


def test_devscore_adds_the_extension_for_its_checks_only(
    cli: CliRunner, calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, PlainDataset)
    plain_out, plain = devscore(cli)
    use(monkeypatch, ExtendedDataset)
    out, extended = devscore(cli)
    assert calls == [len(BASE.clips), len(ALL.clips)]
    assert "extension" not in plain_out
    assert f"extension {EXTENSION}: {len(EXTRA.clips)} clips, 20 scenes, only for R5b" in out
    assert extended["dataset"]["subset"] == f"{plain['dataset']['subset']}+{EXTENSION}"
    before, after = plain["levels"][0]["checks"], extended["levels"][0]["checks"]
    assert {c: before[c] for c in OTHERS} == {c: after[c] for c in OTHERS}
    assert before["R5b"] != after["R5b"]


def test_explicit_clips_take_no_extension(
    cli: CliRunner, calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ExtendedDataset)
    picked = [arg for c in BASE.clips for arg in ("--clips", c.clip_id)]
    out, report = devscore(cli, *picked)
    assert calls == [len(BASE.clips)] and "extension" not in out
    assert "+" not in report["dataset"]["subset"]


def fit_from(cli: CliRunner, settings: Settings, report: VerificationReport, out: Path) -> Any:
    write_verify_run(settings, out.stem, report)
    result = cli(*FAST, "calibrate", "fit", "--dataset", DATASET, "--from-run", out.stem,
                 "--out", str(out))  # fmt: skip
    assert result.code == 0, result.err
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_fit_from_a_run_scopes_the_extension_clips_it_holds(
    cli: CliRunner, settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use(monkeypatch, ExtendedDataset)
    plain = fit_from(cli, settings, BASE, tmp_path / "base.json")
    extended = fit_from(cli, settings, ALL, tmp_path / "all.json")
    assert "+" not in plain["dataset"]["subset"]
    assert extended["dataset"]["subset"].endswith(f"+{EXTENSION}")
    before, after = plain["strata"][0]["checks"], extended["strata"][0]["checks"]
    assert {c: before[c] for c in OTHERS} == {c: after[c] for c in OTHERS}
    assert (before["R5b"]["status"], after["R5b"]["status"]) == ("no_negatives", "ok")


def test_an_extension_missing_from_the_cache_stops_the_run(
    cli: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(monkeypatch, ExtendedDataset)
    monkeypatch.setattr(app_calibration, "_raw_runner", lambda session: _UncachedRunner([]))
    monkeypatch.setattr(app_calibration, "_input_source", lambda session, dataset, level: None)
    result = cli(*FAST, "calibrate", "dev-score", "--dataset", DATASET, "--level", "oracle")
    assert result.code != 0
    assert f"extension {EXTENSION}: {len(EXTRA.clips)} of {len(EXTRA.clips)} clips" in result.err
    assert f"extract run --dataset {DATASET} --subset {EXTENSION}" in result.err
    assert f"--set dataset.subsets.{EXTENSION}.checks_only=[]" in result.err
