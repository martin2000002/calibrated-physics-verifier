"""``physverify calibrate`` fit --from-run, publish and show over a fake dataset (AC1, AC11)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest
from calibration_cases import DATASET, synthetic_report
from cli_runner import CliRunner

from physverify.calibration.loader import load_calibration, projection_name
from physverify.cli import context
from physverify.cli.commands import calibrate
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.verification.runner import CALIBRATED, write_report

EXIT_USAGE = 2
RUN_ID = "raw-oracle"
REPORT = synthetic_report(40)
FAST = ("--set", "calibration.bootstrap=100")
STRICT = ("--set", "calibration.require_calibrated_artifact=true")


class CalibFixtureDataset:
    """The manifests of the synthetic report; no video and no ground truth."""

    name: ClassVar[str] = DATASET

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def clips(self, subset: str | None = None) -> Sequence[ClipManifest]:
        return REPORT.clips

    def video(self, clip: ClipManifest) -> VideoSource:
        raise NotImplementedError

    def ground_truth(self, clip: ClipManifest) -> GroundTruthStore | None:
        return None

    def fetch(self, subset: str | None, dest: Path) -> None:
        return None


@pytest.fixture(autouse=True)
def fixture_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    kinds: Mapping[str, Any] = context.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {**kinds, "datasets": lambda: {**datasets(), DATASET: CalibFixtureDataset}}
    monkeypatch.setattr(context, "PLUGIN_KINDS", MappingProxyType(patched))


@pytest.fixture
def settings(config_dir: Path, environ: dict[str, str]) -> Settings:
    return load_settings(config_dir, environ=environ)


@pytest.fixture
def raw_run(settings: Settings) -> str:
    folder = settings.paths.runs_dir / RUN_ID
    folder.mkdir(parents=True)
    write_report(folder, REPORT)
    return RUN_ID


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(cli: CliRunner, raw_run: str, out: Path) -> None:
    result = cli(
        *FAST, "calibrate", "fit", "--dataset", DATASET, "--from-run", raw_run, "--out", str(out)
    )
    assert result.code == 0, result.err
    assert f"artifact: {out}" in result.out


def test_the_command_is_registered_and_implemented() -> None:
    assert calibrate.IMPLEMENTED and calibrate.OWNER == "A8"
    assert "calibrate" in context.PLUGIN_KINDS["commands"]()


def test_fit_from_a_raw_run_writes_the_artifact_and_projection(
    cli: CliRunner, raw_run: str, tmp_path: Path
) -> None:
    out = tmp_path / "cal" / "calibration.json"
    fit(cli, raw_run, out)
    calibration = load_calibration(out)
    stratum = calibration.stratum(InputLevel.ORACLE, calibration.artifact.strata[0].profile)
    assert stratum.model.checks["R1"].status == "ok"
    assert stratum.model.checks["R3a"].status == "no_positives"
    assert stratum.model.checks["R5b"].status == "no_negatives"
    assert calibration.artifact.dataset.n_scenes == 40
    projection = out.parent / projection_name(InputLevel.ORACLE, stratum.profile)
    assert set(json.loads(projection.read_text(encoding="utf-8"))["checks"]) == {"R1"}


def test_two_fits_differ_only_in_time_and_run_id(
    cli: CliRunner, raw_run: str, tmp_path: Path
) -> None:
    """AC2 through the command."""
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    fit(cli, raw_run, first)
    fit(cli, raw_run, second)

    def stable(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if k not in ("created_utc", "run_id")}

    assert stable(first) == stable(second)


def test_publish_refuses_an_artifact_that_is_not_calibrated(
    cli: CliRunner, raw_run: str, tmp_path: Path, settings: Settings
) -> None:
    out = tmp_path / "calibration.json"
    fit(cli, raw_run, out)
    result = cli(*STRICT, "calibrate", "publish", "--level", "oracle", "--artifact", str(out))
    assert result.code == EXIT_USAGE and "--force" in result.err
    assert "not 'calibrated'" in result.err
    assert not settings.paths.thresholds.exists()


def test_publish_writes_the_thresholds_verify_reads(
    cli: CliRunner, raw_run: str, settings: Settings
) -> None:
    out = settings.paths.calibration_dir / "calibration.json"
    fit(cli, raw_run, out)
    before = sha256(out)
    result = cli(
        *STRICT, "calibrate", "publish", "--level", "oracle", "--artifact", str(out), "--force"
    )
    assert result.code == 0, result.err
    assert "warning: publishing a stratum that is not usable" in result.err
    published = json.loads(settings.paths.thresholds.read_text(encoding="utf-8"))
    assert published["status"] == CALIBRATED
    assert published["artifact"]["sha256"] == before
    assert published["artifact"]["path"] == "calibration.json"
    shown = cli("calibrate", "show", "--artifact", str(out))
    assert shown.code == 0 and "R1 ok tau" in shown.out
    assert sha256(out) == before


def test_publish_follows_the_consumers_gate(
    cli: CliRunner, raw_run: str, tmp_path: Path, settings: Settings
) -> None:
    """Without require_calibrated_artifact only the stratum is required (0b.2): the fixture is
    ``insufficient`` but its oracle stratum is usable, as H2 and H3 would see it."""
    out = tmp_path / "calibration.json"
    fit(cli, raw_run, out)
    assert load_calibration(out).status == "insufficient"
    result = cli("calibrate", "publish", "--level", "oracle", "--artifact", str(out))
    assert result.code == 0, result.err
    assert "warning" not in result.err
    assert settings.paths.thresholds.is_file()


def test_publish_rejects_a_stale_artifact_even_with_force(
    cli: CliRunner, raw_run: str, tmp_path: Path
) -> None:
    out = tmp_path / "calibration.json"
    fit(cli, raw_run, out)
    stale = ("--set", "checks.R2.friction=0.9")
    result = cli(
        *stale, "calibrate", "publish", "--level", "oracle", "--artifact", str(out), "--force"
    )
    assert result.code == EXIT_USAGE and "refit" in result.err


def test_two_runs_at_one_level_are_refused(
    cli: CliRunner, raw_run: str, settings: Settings
) -> None:
    folder = settings.paths.runs_dir / "raw-oracle-b"
    folder.mkdir(parents=True)
    write_report(folder, REPORT)
    result = cli(
        "calibrate",
        "fit",
        "--dataset",
        DATASET,
        "--from-run",
        raw_run,
        "--from-run",
        "raw-oracle-b",
    )
    assert result.code == EXIT_USAGE and "pass one per level" in result.err


@pytest.mark.parametrize("command", ["evaluate", "publish", "show"])
def test_a_missing_artifact_is_a_usage_error(cli: CliRunner, tmp_path: Path, command: str) -> None:
    extra = ("--level", "oracle") if command == "publish" else ()
    result = cli("calibrate", command, *extra, "--artifact", str(tmp_path / "absent.json"))
    assert result.code == EXIT_USAGE


def test_a_policy_decided_run_is_refused(cli: CliRunner, settings: Settings) -> None:
    from dataclasses import replace

    from physverify.schema.checks import Verdict

    decided = replace(REPORT, results=(replace(REPORT.results[0], verdict=Verdict.PASS),))
    folder = settings.paths.runs_dir / "decided"
    folder.mkdir(parents=True)
    write_report(folder, decided)
    result = cli("calibrate", "fit", "--dataset", DATASET, "--from-run", "decided")
    assert result.code != 0
    assert "--policy raw" in result.err + result.out
