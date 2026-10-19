"""ADR 0015 1.4: the H1 primary scene view is a report filter next to the all-scenes strata."""

from __future__ import annotations

from pathlib import Path

import pytest
from calibration_cases import CHECKS, config, dataset_ref, fitted_artifact, synthetic_report
from records import error_profile

from physverify.calibration.collect import level_data
from physverify.calibration.evaluate import (
    EvaluationInputs,
    EvaluationMeta,
    PrimaryScope,
    evaluate,
)
from physverify.calibration.loader import parse_calibration
from physverify.calibration.report import evaluation_markdown
from physverify.calibration.scenes import SceneScope, expand, h1_scope, in_scope, scene_index
from physverify.calibration.schema import CalibrationArtifact, EvaluationReport
from physverify.calibration.settings import active_checks_hash
from physverify.cli.context import PLUGIN_KINDS
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.errors import ConfigError
from physverify.core.hashing import error_profile_hash
from physverify.extraction.cache import plan_entry_filter, plan_params_hashes, profile_keys
from physverify.extraction.plan import resolve_plan
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.schema.stages import Stage

PRIMARY = ("s000", "s011-s039")


def test_specs_expand_to_inclusive_indices() -> None:
    assert expand(PRIMARY) == frozenset({0, *range(11, 40)})
    assert expand(()) == frozenset()


@pytest.mark.parametrize("spec", ["000", "s3-s1", "s1-", "s1-s2-s3", "x001"])
def test_bad_specs_are_config_errors(spec: str) -> None:
    with pytest.raises(ConfigError):
        SceneScope(primary_scenes=(spec,))


def test_scene_ids_match_by_trailing_index() -> None:
    assert scene_index("floor-crossing-s012") == 12
    assert scene_index("s012") is None
    keep = expand(PRIMARY)
    assert in_scope("pair-collision-s000", keep)
    assert not in_scope("pair-collision-s005", keep)
    assert not in_scope("pair-collision-s040", keep)


def test_the_shipped_scope_is_the_adr_one(repo_root: Path) -> None:
    settings = load_settings(repo_root / "configs", environ={})
    assert h1_scope(settings).primary_scenes == PRIMARY


def _hashes(settings: Settings) -> tuple[str, tuple[str, ...], dict[Stage, str]]:
    """``checks_hash``, the error-profile hash and the params hash of every stage of the plan."""
    plan = resolve_plan(settings, PLUGIN_KINDS["providers"](), dataset="kubric_synthetic")
    accept = plan_entry_filter(plan)
    profile = error_profile()
    stages = tuple(
        error_profile_hash(profile, profile_keys(step.provider, plan.names), accept)
        for step in plan.stages
    )
    return active_checks_hash(settings), stages, plan_params_hashes(plan.stages)


def test_the_scope_changes_neither_checks_hash_nor_the_error_profile_hash(repo_root: Path) -> None:
    """A report filter only (ADR 0015 1.4): the artifact, decide and the cache keys keep."""
    configs = repo_root / "configs"
    shipped = load_settings(configs, environ={}, profile="precise")
    moved = load_settings(
        configs, environ={}, profile="precise", overrides=['h1_scope.primary_scenes=["s001"]']
    )
    assert h1_scope(moved).primary_scenes != h1_scope(shipped).primary_scenes
    assert _hashes(moved) == _hashes(shipped)


@pytest.fixture(scope="module")
def artifact() -> CalibrationArtifact:
    return fitted_artifact()


def _inputs() -> EvaluationInputs:
    calib = synthetic_report(40)
    heldout = synthetic_report(30, seed=7, split=Split.EVAL, prefix="f1-s")
    both = type(calib)(
        calib.results + heldout.results, calib.table + heldout.table, calib.clips + heldout.clips
    )
    real = synthetic_report(20, seed=9, split=Split.EVAL, prefix="r")
    real_data = level_data(InputLevel.ORACLE, real, CHECKS, config(), split=None)
    synthetic = level_data(InputLevel.ORACLE, both, CHECKS, config(), split=None)
    return EvaluationInputs(synthetic, {"real_fixture": real_data})


def _meta(primary: PrimaryScope | None) -> EvaluationMeta:
    return EvaluationMeta(
        "2026-10-19T00:00:00Z",
        "eval-a",
        "calibration.json",
        dataset_ref(),
        "fast",
        primary=primary,
    )


def _evaluate(artifact: CalibrationArtifact, primary: PrimaryScope | None) -> EvaluationReport:
    return evaluate(parse_calibration(artifact.to_json()), (_inputs(),), _meta(primary), config())


def _scope() -> PrimaryScope:
    dataset = dataset_ref().model_copy(update={"n_scenes": 20})
    return PrimaryScope(SceneScope(primary_scenes=PRIMARY), dataset)


def test_the_all_scenes_strata_do_not_change(artifact: CalibrationArtifact) -> None:
    without = _evaluate(artifact, None)
    with_primary = _evaluate(artifact, _scope())
    assert without.h1_primary is None
    assert with_primary.strata == without.strata
    assert with_primary.artifact_sha256 == without.artifact_sha256


def test_the_primary_view_keeps_only_its_scenes(artifact: CalibrationArtifact) -> None:
    report = _evaluate(artifact, _scope())
    assert report.h1_primary is not None
    (primary,) = report.h1_primary.strata
    (whole,) = report.strata
    r1 = next(c for c in primary.checks if c.check_id == "R1")
    assert r1.fpr_clean.n == 20
    assert next(c for c in whole.checks if c.check_id == "R1").fpr_clean.n == 30
    assert primary.real == () and primary.real_clip_flags == ()
    assert {k.scene_id for k in primary.scene_kendall} <= {f"f1-s{i:03d}" for i in expand(PRIMARY)}


def test_an_empty_scope_gives_no_view(artifact: CalibrationArtifact) -> None:
    empty = PrimaryScope(SceneScope(), dataset_ref())
    assert _evaluate(artifact, empty).h1_primary is None


def test_the_report_round_trips_and_renders(artifact: CalibrationArtifact) -> None:
    report = _evaluate(artifact, _scope())
    assert EvaluationReport.model_validate_json(report.to_json()) == report
    markdown = evaluation_markdown(report)
    assert "## H1 primary (scenes s000, s011-s039) vs all scenes" in markdown
    assert "FPR clean primary" in markdown and "Primary: " in markdown
    assert "H1 primary" not in evaluation_markdown(_evaluate(artifact, None))
