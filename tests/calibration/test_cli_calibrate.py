"""``physverify calibrate`` fit --from-run, publish and show over a fake dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest
from calibration_cases import DATASET, synthetic_report
from cli_runner import CliRunner
from cli_tree import command_modules
from steps_sh import steps_read

from physverify.app import calibration as app_calibration
from physverify.app import plugins
from physverify.calibration.artifact_io import load_calibration, projection_name
from physverify.config.models import Settings
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.schema.checks import InputLevel
from physverify.schema.clip import ClipManifest
from physverify.verification.runner import CALIBRATED, write_report

EXIT_USAGE = 2
RUN_ID = "raw-oracle"
REPORT = synthetic_report(40)
FAMILY_REPORT = synthetic_report(40, prefix="family-s")
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
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {**kinds, "datasets": lambda: {**datasets(), DATASET: CalibFixtureDataset}}
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))


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


def test_the_command_is_registered() -> None:
    assert "calibrate" in command_modules()


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
    ``insufficient`` but its oracle stratum is usable, as the human-agreement study and the
    generator comparison would see it."""
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


class _ReportRunner:
    def run(self, source: object, clips: Sequence[ClipManifest]) -> object:
        return REPORT


def _evaluate_cli(cli: CliRunner, raw_run: str, tmp_path: Path, *extra: str) -> dict[str, Any]:
    out = tmp_path / "cal" / "calibration.json"
    fit(cli, raw_run, out)
    result = cli(*FAST, *extra, "calibrate", "evaluate", "--dataset", DATASET, "--no-real",
                 "--artifact", str(out))  # fmt: skip
    assert result.code == 0, result.err
    path = Path(result.out.split("evaluation: ")[1].split()[0])
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.fixture
def family_run(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> str:
    """A raw run whose scene ids end in ``-sNNN`` like Kubric's; ``calibrate evaluate`` scores
    that report instead of running the checks."""
    monkeypatch.setattr(sys.modules[__name__], "REPORT", FAMILY_REPORT)
    monkeypatch.setattr(app_calibration, "_raw_runner", lambda session: _ReportRunner())
    monkeypatch.setattr(app_calibration, "_input_source", lambda session, dataset, level: None)
    folder = settings.paths.runs_dir / "family"
    folder.mkdir(parents=True)
    write_report(folder, FAMILY_REPORT)
    return "family"


def test_evaluate_adds_the_primary_monotonicity_view_of_the_configured_scenes(
    cli: CliRunner, family_run: str, tmp_path: Path
) -> None:
    """Rule 1.4 through the command: s011-s039 of the 40 scenes (s000 is left out because it
    was looked at during the campaign)."""
    data = _evaluate_cli(cli, family_run, tmp_path)
    primary = data["h1_primary"]
    assert primary["scenes"] == ["s011-s039"]
    assert primary["dataset"]["n_scenes"] == 29
    assert data["dataset"]["n_scenes"] == 40
    assert primary["dataset"]["n_clips"] * 40 == data["dataset"]["n_clips"] * 29


def test_evaluate_without_primary_scenes_has_no_primary_monotonicity_view(
    cli: CliRunner, family_run: str, tmp_path: Path
) -> None:
    data = _evaluate_cli(cli, family_run, tmp_path, "--set", "h1_scope.primary_scenes=[]")
    assert data["h1_primary"] is None


def test_the_runners_read_the_evaluation_calibrate_evaluate_prints(
    cli: CliRunner, family_run: str, tmp_path: Path, settings: Settings
) -> None:
    out = tmp_path / "cal" / "calibration.json"
    fit(cli, family_run, out)
    result = cli(*FAST, "calibrate", "evaluate", "--dataset", DATASET, "--no-real",
                 "--artifact", str(out))  # fmt: skip
    assert result.code == 0, result.err
    log = tmp_path / "evaluate.log"
    log.write_text(result.out, encoding="utf-8")
    path = Path(steps_read(log, "evaluation_of", str(log)))
    assert path.is_file() and path.parent.parent == settings.paths.runs_dir


def _snapshot(*folders: Path) -> dict[Path, tuple[str, int]]:
    return {
        path: (sha256(path), path.stat().st_mtime_ns)
        for folder in folders
        if folder.is_dir()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def test_evaluate_writes_only_its_run_and_repeats_the_artifacts_tau(
    cli: CliRunner, family_run: str, tmp_path: Path, settings: Settings
) -> None:
    """The extra held-out monotonicity evaluation adds a run; the artifact and calibration_dir
    keep."""
    out = tmp_path / "cal" / "calibration.json"
    fit(cli, family_run, out)
    before = _snapshot(out.parent, settings.paths.calibration_dir)
    result = cli(*FAST, "calibrate", "evaluate", "--dataset", DATASET, "--no-real",
                 "--artifact", str(out))  # fmt: skip
    assert result.code == 0, result.err
    assert _snapshot(out.parent, settings.paths.calibration_dir) == before
    path = Path(result.out.split("evaluation: ")[1].split()[0])
    assert path.parent.parent == settings.paths.runs_dir
    data = json.loads(path.read_text(encoding="utf-8"))
    fitted = json.loads(out.read_text(encoding="utf-8"))["strata"][0]["checks"]
    pooled = {m["check_id"]: m["monotonicity"] for m in data["strata"][0]["monotonicity"]}
    assert pooled and all(pooled[c] == fitted[c]["monotonicity"] for c in pooled)
