"""ADR 0015 1.4: the H1 primary scene view is a report filter next to the all-scenes strata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from calibration_cases import (
    CHECKS,
    config,
    dataset_ref,
    fit_meta,
    fitted_artifact,
    settings,
    synthetic_level,
    synthetic_report,
)
from records import error_profile
from wiring import active_checks_hash, resolve_plan

from physverify.app.plugins import PLUGIN_KINDS
from physverify.calibration.artifact import (
    CALIBRATION_VERSION,
    CalibrationArtifact,
    EvaluationReport,
    Monotonicity,
)
from physverify.calibration.artifact_io import parse_calibration
from physverify.calibration.collect import level_data
from physverify.calibration.evaluate import (
    EvaluationInputs,
    EvaluationMeta,
    PrimaryScope,
    evaluate,
)
from physverify.calibration.fit import FitInputs, fit_calibration
from physverify.calibration.monotonicity import LevelDelta, monotonicity
from physverify.calibration.report import evaluation_markdown
from physverify.config.loader import load_settings
from physverify.config.models import Settings
from physverify.core.cache_keys import error_profile_hash
from physverify.core.digests import text_sha256
from physverify.core.errors import ConfigError
from physverify.extraction.cache import plan_entry_filter, plan_params_hashes, profile_keys
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.schema.stages import Stage
from physverify.synth.scene_scope import SceneScope, expand, h1_scope, in_scope, scene_index
from physverify.verification.runner import VerificationReport

PRIMARY = ("s000", "s011-s039")
SHIPPED = ("s011-s039",)
"""Amendment 2 section 6 (plan-congelado Q2): s000 was looked at during the campaign."""


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
    assert h1_scope(settings).primary_scenes == SHIPPED


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


# --- the pooled tau_b of D8 in both views (ADR 0015 1.4, rule of ADR 0009 change 4) -----------


def _fit_on(report: VerificationReport) -> CalibrationArtifact:
    level = synthetic_level(report, InputLevel.ORACLE)
    return fit_calibration(FitInputs((level,), CHECKS, fit_meta()), config())


@pytest.fixture(scope="module")
def family() -> tuple[CalibrationArtifact, EvaluationInputs]:
    """Calib scenes named like Kubric's (``-sNNN``) so that the primary scope selects them."""
    calib = synthetic_report(40, prefix="fam-s")
    heldout = synthetic_report(40, seed=7, split=Split.EVAL, prefix="held-s")
    both = type(calib)(
        calib.results + heldout.results, calib.table + heldout.table, calib.clips + heldout.clips
    )
    synthetic = level_data(InputLevel.ORACLE, both, CHECKS, config(), split=None)
    return _fit_on(calib), EvaluationInputs(synthetic, {})


def _json(model: Monotonicity) -> object:
    return json.loads(model.to_json())


def _family_report(family: tuple[CalibrationArtifact, EvaluationInputs]) -> EvaluationReport:
    artifact, inputs = family
    calibration = parse_calibration(artifact.to_json())
    return evaluate(calibration, (inputs,), _meta(_scope()), config(bootstrap=50, seed=3))


def test_the_all_scenes_tau_is_the_artifacts_own(
    family: tuple[CalibrationArtifact, EvaluationInputs],
) -> None:
    """Same value, interval and seed as the fit, even with other evaluation settings."""
    artifact, _ = family
    report = _family_report(family)
    (whole,) = report.strata
    fitted = artifact.strata[0].checks
    by_id = {m.check_id: m.monotonicity for m in whole.monotonicity}
    assert set(by_id) == {c for c, f in fitted.items() if f.monotonicity.n_pairs > 0} == {"R1"}
    for check_id, pooled in by_id.items():
        assert _json(pooled) == _json(fitted[check_id].monotonicity)
    assert {m.split for m in whole.monotonicity} == {"calib"}


def test_the_primary_tau_uses_only_the_primary_scenes(
    family: tuple[CalibrationArtifact, EvaluationInputs],
) -> None:
    report = _family_report(family)
    assert report.h1_primary is not None
    (primary,) = report.h1_primary.strata
    (whole,) = report.strata
    (r1_primary,) = primary.monotonicity
    (r1_all,) = whole.monotonicity
    assert r1_all.monotonicity.n_scenes == 40
    assert r1_primary.monotonicity.n_scenes == len(expand(PRIMARY)) == 30
    assert r1_primary.monotonicity.n_pairs * 40 == r1_all.monotonicity.n_pairs * 30
    markdown = evaluation_markdown(report)
    assert "Pooled Kendall tau_b of the calib clips (D8), primary vs all scenes:" in markdown
    assert "median delta by level, primary (descriptive)" in markdown


def test_h1_pass_is_the_lower_bound_only() -> None:
    """Medians that dip at one level do not condition h1_pass (ADR 0009 change 4)."""
    steps = {0: 0.0, 1: 0.2, 2: 0.1, 3: 0.5}
    pairs = [LevelDelta(f"s{i:03d}", lv, d) for i in range(20) for lv, d in steps.items()]
    result = monotonicity(pairs, settings())
    assert result.monotone_medians is False
    assert result.ci.low is not None and result.ci.low > 0.0
    assert result.h1_pass is True


def test_a_1_3_0_evaluation_still_reads(
    family: tuple[CalibrationArtifact, EvaluationInputs],
) -> None:
    data = json.loads(_family_report(family).to_json())
    data["version"] = "1.3.0"
    for stratum in (*data["strata"], *data["h1_primary"]["strata"]):
        del stratum["monotonicity"]
    old = EvaluationReport.model_validate_json(json.dumps(data))
    assert old.strata[0].monotonicity == ()
    assert old.h1_primary is not None and old.h1_primary.strata[0].monotonicity == ()


def test_a_1_3_0_artifact_loads_strictly_and_is_not_rewritten(
    family: tuple[CalibrationArtifact, EvaluationInputs],
) -> None:
    artifact, inputs = family
    text = artifact.to_json().replace(f'"version": "{CALIBRATION_VERSION}"', '"version": "1.3.0"')
    assert '"version": "1.3.0"' in text
    calibration = parse_calibration(text)
    assert calibration.version == "1.3.0"
    assert calibration.artifact.to_json() == text
    report = evaluate(calibration, (inputs,), _meta(_scope()), config())
    assert report.artifact_sha256 == text_sha256(text)
    assert report.strata[0].monotonicity
