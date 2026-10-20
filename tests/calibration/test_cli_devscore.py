"""``physverify calibrate devscore`` over a fake calib dataset (WP-O AC7, AC8, C3)."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest
from calibration_cases import DATASET, synthetic_report
from cli_runner import CliRunner

from physverify.calibration.loader import load_calibration
from physverify.cli import context
from physverify.cli.commands import calibrate
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.schema.clip import ClipManifest, Split
from physverify.verification.runner import VerificationReport

EXIT_USAGE = 2
CALIB = synthetic_report(24)
HELD = synthetic_report(4, seed=7, split=Split.EVAL, prefix="e")
CLIPS = CALIB.clips + HELD.clips
FAST = ("--set", "calibration.bootstrap=30", "--set", "calibration.folds=3")


class DevFixtureDataset:
    name: ClassVar[str] = DATASET

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return CLIPS

    def video(self, clip: ClipManifest) -> VideoSource:
        raise NotImplementedError

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


class _CalibRunner:
    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    def run(self, source: object, clips: Sequence[ClipManifest]) -> VerificationReport:
        self.calls.append(len(clips))
        return CALIB


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    kinds: Mapping[str, Any] = context.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {**kinds, "datasets": lambda: {**datasets(), DATASET: DevFixtureDataset}}
    monkeypatch.setattr(context, "PLUGIN_KINDS", MappingProxyType(patched))
    found: list[int] = []
    monkeypatch.setattr(calibrate, "_raw_runner", lambda session: _CalibRunner(found))
    monkeypatch.setattr(calibrate, "_source", lambda session, dataset, level: None)
    return found


@pytest.fixture
def settings(config_dir: Path, environ: dict[str, str]) -> Settings:
    return load_settings(config_dir, environ=environ)


def _snapshot(*paths: Path) -> dict[Path, tuple[str, int]]:
    files = [p for path in paths if path.exists() for p in [path, *path.rglob("*")] if p.is_file()]
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files}


def test_an_eval_clip_is_refused_before_any_check_runs(cli: CliRunner, calls: list[int]) -> None:
    """AC7."""
    held = HELD.clips[0].clip_id
    result = cli(*FAST, "calibrate", "devscore", "--dataset", DATASET, "--level", "oracle",
                 "--clips", CALIB.clips[0].clip_id, "--clips", held)  # fmt: skip
    assert result.code == EXIT_USAGE and held in result.err
    assert calls == []


def test_bootstrap_zero_is_refused_before_any_check_runs(cli: CliRunner, calls: list[int]) -> None:
    result = cli("--set", "calibration.bootstrap=0", "calibrate", "devscore", "--dataset", DATASET)
    assert result.code == EXIT_USAGE and "bootstrap" in result.err
    assert calls == []


def test_devscore_writes_only_its_run_and_out(
    cli: CliRunner, calls: list[int], settings: Settings, tmp_path: Path
) -> None:
    """AC8 and C3: the run holds devscore.json and the full fit; calibration_dir, the
    thresholds and the error profile keep their listing (sha and mtime)."""
    paths = settings.paths
    paths.calibration_dir.mkdir(parents=True, exist_ok=True)
    (paths.calibration_dir / "calibration.json").write_text("{}\n", encoding="utf-8")
    before = _snapshot(paths.calibration_dir, paths.thresholds, paths.error_profile)
    out = tmp_path / "copy" / "devscore.json"
    result = cli(*FAST, "calibrate", "devscore", "--dataset", DATASET, "--level", "oracle",
                 "--out", str(out))  # fmt: skip
    assert result.code == 0, result.err
    assert calls == [len(CALIB.clips)]
    assert _snapshot(paths.calibration_dir, paths.thresholds, paths.error_profile) == before
    run_dir = Path(result.out.split("run: ")[1].split()[0])
    assert run_dir.parent == paths.runs_dir
    written = (run_dir / "devscore.json").read_text(encoding="utf-8")
    assert out.read_text(encoding="utf-8") == written
    data = json.loads(written)
    assert data["schema"] == "physverify.devscore" and data["dataset"]["split"] == "calib"
    assert "not a held-out result" in data["scope_note"]
    fit = load_calibration(run_dir / "calibration.json")
    assert [lv["profile"] for lv in data["levels"]] == [s.profile for s in fit.artifact.strata]
    assert re.search(r"oracle/fast: 1/\d+ meet H1 and J_cv\^lo > 0 \(R1\)", result.out)
