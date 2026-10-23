"""``physverify calibrate dev-score`` over a fake calib dataset."""

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
from steps_sh import steps_read
from wiring import checks

from physverify.app import calibration as app_calibration
from physverify.app import plugins
from physverify.calibration.artifact_io import load_calibration
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest, Split
from physverify.verification.runner import VerificationReport

EXIT_USAGE = 2
DEVELOPMENT_METRIC_LINES = re.compile("^(note: |oracle/|perceived/|  R|selection )")
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
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {**kinds, "datasets": lambda: {**datasets(), DATASET: DevFixtureDataset}}
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))
    found: list[int] = []
    monkeypatch.setattr(app_calibration, "_raw_runner", lambda session: _CalibRunner(found))
    monkeypatch.setattr(app_calibration, "_input_source", lambda session, dataset, level: None)
    return found


@pytest.fixture
def settings(config_dir: Path, environ: dict[str, str]) -> Settings:
    return load_settings(config_dir, environ=environ)


def _snapshot(*paths: Path) -> dict[Path, tuple[str, int]]:
    files = [p for path in paths if path.exists() for p in [path, *path.rglob("*")] if p.is_file()]
    return {p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in files}


def test_an_eval_clip_is_refused_before_any_check_runs(cli: CliRunner, calls: list[int]) -> None:
    held = HELD.clips[0].clip_id
    result = cli(*FAST, "calibrate", "dev-score", "--dataset", DATASET, "--level", "oracle",
                 "--clips", CALIB.clips[0].clip_id, "--clips", held)  # fmt: skip
    assert result.code == EXIT_USAGE and held in result.err
    assert calls == []


def test_bootstrap_zero_is_refused_before_any_check_runs(cli: CliRunner, calls: list[int]) -> None:
    result = cli("--set", "calibration.bootstrap=0", "calibrate", "dev-score", "--dataset", DATASET)
    assert result.code == EXIT_USAGE and "bootstrap" in result.err
    assert calls == []


def test_devscore_writes_only_its_run_and_out(
    cli: CliRunner, calls: list[int], settings: Settings, tmp_path: Path
) -> None:
    """The run holds devscore.json and the full fit; calibration_dir, the
    thresholds and the error profile keep their listing (sha and mtime)."""
    paths = settings.paths
    paths.calibration_dir.mkdir(parents=True, exist_ok=True)
    (paths.calibration_dir / "calibration.json").write_text("{}\n", encoding="utf-8")
    before = _snapshot(paths.calibration_dir, paths.thresholds, paths.error_profile)
    out = tmp_path / "copy" / "devscore.json"
    result = cli(*FAST, "calibrate", "dev-score", "--dataset", DATASET, "--level", "oracle",
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
    assert re.search(r"oracle/fast: 1/\d+ meet monotonicity and J_cv\^lo > 0 \(R1\)", result.out)


def test_the_runners_keep_the_development_metric_lines(
    cli: CliRunner, calls: list[int], tmp_path: Path
) -> None:
    """run_perception_iteration.sh step 5: run_of finds the run and its grep keeps the metric
    lines."""
    result = cli(*FAST, "calibrate", "dev-score", "--dataset", DATASET, "--level", "oracle")
    assert result.code == 0, result.err
    log = tmp_path / "devscore.log"
    log.write_text(result.out, encoding="utf-8")
    assert (Path(steps_read(log, "run_of")) / "devscore.json").is_file()
    kept = [line for line in result.out.splitlines() if DEVELOPMENT_METRIC_LINES.match(line)]
    assert kept[0].startswith("note: ") and kept[1].startswith("oracle/fast: ")
    assert [line.split(":")[0] for line in kept[2:]] == [f"  {c}" for c in sorted(checks())]
    assert InputLevel.PERCEIVED.value == "perceived"
