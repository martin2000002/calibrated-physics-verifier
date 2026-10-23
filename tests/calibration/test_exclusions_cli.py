"""``physverify exclusions``: reading the R2 arms back, the replace rule and the command."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pytest
from calibration_cases import CHECKS, clip, result, synthetic_report
from calibration_cases import record as perturbation
from cli_runner import CliRunner
from exclusion_cases import PRECISE, R2_PERT, evaluation, r5a_check, real_fpr, record, rules
from steps_sh import steps_read
from stratum_cases import artifact_of, check_calibration, stratum_model
from wiring import active_checks_hash, checks

from physverify.app import plugins
from physverify.calibration.artifact_io import default_artifact_path, parse_calibration
from physverify.cli.commands import exclusions
from physverify.cli.commands.exclusions import (
    arm_run,
    families_without_target,
    r2_clip_ids,
    replacement,
)
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.errors import UsageError
from physverify.datasets._base import GroundTruthStore, VideoSource
from physverify.finetune.settings import load_finetune
from physverify.schema.checks import InputLevel, Verdict
from physverify.schema.clip import ClipManifest
from physverify.schema.scene import MagnitudeKind
from physverify.verification.exclusion_record import ExclusionInputError, parse_record
from physverify.verification.runner import (
    CHECKS_HASH_FILE,
    RowStatus,
    TableRow,
    VerificationReport,
    write_report,
)
from physverify.verification.scope import CheckScope, check_scope

FAMILY_PREFIX = "f4_stack-s"
PERTURBATION = "r1_penetration"


def _report(level: InputLevel = InputLevel.PERCEIVED, detector: str = "grounding_dino_base"):  # type: ignore[no-untyped-def]
    report = synthetic_report(4, prefix=FAMILY_PREFIX, level=level)
    results = tuple(
        dataclasses.replace(r, provenance={**r.provenance, "detect": detector})
        for r in report.results
    )
    return dataclasses.replace(report, results=results)


def _run(  # type: ignore[no-untyped-def]
    root: Path, name: str, report, profile: str = "precise", checks_hash: str | None = None
) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    write_report(folder, report)
    run = {"profile": profile, "commit": "c0"}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    if checks_hash is not None:
        (folder / CHECKS_HASH_FILE).write_text(checks_hash + "\n", encoding="utf-8")
    return folder


def test_the_command_is_registered() -> None:
    assert exclusions.NAME == "exclusions"


def test_r2_clips_are_the_clean_and_r2_clips_of_the_family() -> None:
    cfg = rules().r2.model_copy(update={"perturbation": PERTURBATION})
    clips = synthetic_report(2, prefix=FAMILY_PREFIX).clips + synthetic_report(1).clips
    ids = r2_clip_ids(clips, cfg)
    assert ids == sorted(
        f"{FAMILY_PREFIX}{s:03d}__{name}"
        for s in range(2)
        for name in ("none", *(f"{PERTURBATION}_L{lv}" for lv in (1, 2, 3)))
    )


def test_an_arm_is_read_back_with_its_detector_profile_and_hash(tmp_path: Path) -> None:
    report = _report()
    folder = _run(tmp_path, "base", report)
    arm = arm_run(folder, report.clips, CHECKS, frozenset(), rules())
    assert arm.detectors == {"grounding_dino_base"}
    assert (arm.run_id, arm.profile, arm.commit, arm.checks_hash) == ("base", "precise", "c0", "")
    assert len(arm.rows) == len(report.table)
    again = arm_run(_run(tmp_path, "copy", report), report.clips, CHECKS, frozenset(), rules())
    assert again.report_sha256 == arm.report_sha256


def test_an_arm_at_another_level_is_refused(tmp_path: Path) -> None:
    report = _report(InputLevel.ORACLE)
    folder = _run(tmp_path, "oracle", report)
    with pytest.raises(ExclusionInputError, match="not perceived"):
        arm_run(folder, report.clips, CHECKS, frozenset(), rules())


def test_a_record_is_replaced_only_on_request_and_bound_to_the_old_one() -> None:
    old, fresh = record(("R1",)), record(("R2",))
    assert replacement(None, fresh, replace=False, label="x") == fresh
    assert replacement(old.to_json(), old, replace=False, label="x") is None
    with pytest.raises(UsageError, match="--replace"):
        replacement(old.to_json(), fresh, replace=False, label="x")
    new = replacement(old.to_json(), fresh, replace=True, label="x")
    assert new is not None and new.replaces_sha256 == old.sha256
    assert new.sealed().sha256 == new.sha256
    assert replacement(new.to_json(), fresh, replace=False, label="x") is None


def test_the_families_left_without_a_target_are_reported(
    config_dir: Path, environ: dict[str, str]
) -> None:
    cfg = load_finetune(load_settings(config_dir, environ=environ))
    gap = families_without_target(cfg, CheckScope(exclude_checks=("R1",)))
    assert "stack" not in gap(frozenset())
    assert "stack" in gap(frozenset({"R2"}))


def test_decide_without_an_evaluation_is_a_usage_error(cli: CliRunner, tmp_path: Path) -> None:
    result = cli(
        "exclusions",
        "decide",
        "--evaluation",
        str(tmp_path / "missing.json"),
        "--r2-base-run",
        "a",
        "--r2-tiny-run",
        "b",
    )
    assert result.code == 2


R2_UNIT = "residual_over_weight"
N_SCENES = 20


def _r2_clips() -> tuple[ClipManifest, ...]:
    found: list[ClipManifest] = []
    for s in range(N_SCENES):
        scene = f"{FAMILY_PREFIX}{s:03d}"
        pert = perturbation(R2_PERT, ("R2",), 0.2, R2_UNIT, 1, MagnitudeKind.INJECTED_PARAMETER)
        found += [
            clip(f"{scene}__none", scene),
            clip(f"{scene}__{R2_PERT}_L1", scene, perturbation=pert),
        ]
    return tuple(found)


CLIPS = _r2_clips()


class R2Dataset:
    name: ClassVar[str] = "kubric_synthetic"

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


def _r2_report(detector: str) -> VerificationReport:
    table, results = [], []
    for manifest in CLIPS:
        m = 0.1 if manifest.perturbation is None else 2.0
        table.append(
            TableRow(
                clip_id=manifest.clip_id,
                check_id="R2",
                status=RowStatus.SCORED,
                magnitude=m,
                sigma=0.0,
                verdict=Verdict.UNSCORED,
                abstain_reason=None,
                n_subjects=1,
                n_abstained=0,
                unit=R2_UNIT,
                row_step=1,
            )
        )
        found = result("R2", manifest.clip_id, m)
        results.append(dataclasses.replace(found, unit=R2_UNIT, provenance={"detect": detector}))
    return VerificationReport(results=tuple(results), table=tuple(table), clips=CLIPS)


@pytest.fixture
def world(config_dir: Path, environ: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    kinds: Mapping[str, Any] = plugins.PLUGIN_KINDS
    datasets = kinds["datasets"]
    patched = {**kinds, "datasets": lambda: {**datasets(), R2Dataset.name: R2Dataset}}
    monkeypatch.setattr(plugins, "PLUGIN_KINDS", MappingProxyType(patched))
    settings = load_settings(config_dir, environ=environ)
    checks = {"R2": check_calibration(tau=1.0), "R5a": r5a_check()}
    stratum = stratum_model(checks, InputLevel.PERCEIVED, 0.5, PRECISE)
    text = artifact_of((stratum,), checks_hash=active_checks_hash(settings)).to_json()
    artifact = default_artifact_path(settings)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(text, encoding="utf-8")
    real = [real_fpr("physics_iq", "R2", 0, 54), real_fpr("pisa_real", "R2", 0, 360)]
    report = evaluation(parse_calibration(text, path=artifact), real)
    (settings.paths.runs_dir / "eval").mkdir(parents=True)
    (settings.paths.runs_dir / "eval" / "evaluation.json").write_text(
        report.to_json(), encoding="utf-8"
    )
    checks_hash = active_checks_hash(settings)
    for name, detector in (("base", "grounding_dino_base"), ("tiny", "grounding_dino_tiny")):
        _run(settings.paths.runs_dir, name, _r2_report(detector), checks_hash=checks_hash)
    return settings


def _decide(cli: CliRunner, world: Settings, *extra: str):  # type: ignore[no-untyped-def]
    evaluation_path = world.paths.runs_dir / "eval" / "evaluation.json"
    return cli(
        "exclusions",
        "decide",
        "--evaluation",
        str(evaluation_path),
        "--r2-base-run",
        "base",
        "--r2-tiny-run",
        "tiny",
        *extra,
    )


def test_decide_writes_the_record_check_scope_merges(cli: CliRunner, world: Settings) -> None:
    dry = _decide(cli, world, "--dry-run")
    assert dry.code == 0, dry.err
    target = world.paths.calibration_dir / "exclusions.json"
    assert not target.exists()
    written = _decide(cli, world)
    assert written.code == 0, written.err
    scope = check_scope(world, checks())
    assert scope.record is not None and f"sha256 {scope.record_sha256}" in written.out
    assert scope.record.inputs.r2_base_run == "base"
    r2 = [e for e in scope.record.evidence if e.rule == "1.2"]
    assert r2 and r2[0].decided and not r2[0].excluded
    found = r2[0].inputs
    assert (found["base.n_clean"], found["base.n_positive"]) == (N_SCENES, N_SCENES)
    assert (found["base.fpr"], found["base.tpr"], found["delta_fpr"]) == (0.0, 1.0, 0.0)
    assert target.with_suffix(".md").is_file()
    assert "unchanged" in _decide(cli, world).out


def test_swapped_arms_are_refused(cli: CliRunner, world: Settings) -> None:
    evaluation_path = world.paths.runs_dir / "eval" / "evaluation.json"
    result = cli(
        "exclusions",
        "decide",
        "--evaluation",
        str(evaluation_path),
        "--r2-base-run",
        "tiny",
        "--r2-tiny-run",
        "base",
    )
    assert result.code == 2 and "detectors" in result.err


def test_an_arm_without_its_checks_hash_is_refused(cli: CliRunner, world: Settings) -> None:
    (world.paths.runs_dir / "tiny" / CHECKS_HASH_FILE).unlink()
    result = _decide(cli, world, "--dry-run")
    assert result.code == 2 and "checks_hash" in result.err


def _written(cli: CliRunner, world: Settings) -> Path:
    assert _decide(cli, world).code == 0
    return world.paths.calibration_dir / "exclusions.json"


def test_show_prints_the_record_and_its_sha(cli: CliRunner, world: Settings) -> None:
    target = _written(cli, world)
    sha = parse_record(target.read_text(encoding="utf-8"), "x").sha256
    shown = cli("exclusions", "show")
    assert shown.code == 0, shown.err
    assert f"record: {target} sha256 {sha}" in shown.out
    assert "note:" not in shown.out


def test_the_runners_read_the_hash_check_scope_carries(
    cli: CliRunner, world: Settings, tmp_path: Path
) -> None:
    """Juicio J12: record_sha_of (run_freeze.sh's RECORD_SHA, freeze.json's record_sha) on the
    printed line is the stable hash the H2 pre-registration and run_h2.sh compare."""
    _written(cli, world)
    log = tmp_path / "show.log"
    log.write_text(cli("exclusions", "show").out, encoding="utf-8")
    scope = check_scope(world, checks())
    assert scope.record_sha256 is not None
    assert steps_read(log, "record_sha_of", str(log)) == scope.record_sha256


def test_show_without_a_record_is_a_usage_error(cli: CliRunner, world: Settings) -> None:
    result = cli("exclusions", "show")
    assert result.code == 2 and "exclusions decide" in result.err


def test_show_refuses_an_edited_record(cli: CliRunner, world: Settings) -> None:
    target = _written(cli, world)
    data = json.loads(target.read_text(encoding="utf-8"))
    data["exclude_checks"] = ["R2"]
    target.write_text(json.dumps(data), encoding="utf-8")
    result = cli("exclusions", "show")
    assert result.code == 2 and "edited" in result.err


def test_show_checks_the_active_checks_hash_unless_told(cli: CliRunner, world: Settings) -> None:
    target = _written(cli, world)
    record = parse_record(target.read_text(encoding="utf-8"), "x")
    other = record.model_copy(update={"checks_hash": "f" * 64}).sealed()
    target.write_text(other.to_json(), encoding="utf-8")
    refused = cli("exclusions", "show")
    assert refused.code == 2 and "other check parameters" in refused.err
    shown = cli("exclusions", "show", "--no-checks-hash")
    assert shown.code == 0, shown.err
    assert "not validated" in shown.out and f"sha256 {other.sha256}" in shown.out


def test_show_notes_an_artifact_that_changed(cli: CliRunner, world: Settings) -> None:
    _written(cli, world)
    artifact = default_artifact_path(world)
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    shown = cli("exclusions", "show")
    assert shown.code == 0 and "is not the artifact the record was decided on" in shown.out
