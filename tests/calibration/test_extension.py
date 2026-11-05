"""``calibration.extension``: scenes that count only for some checks (``checks_only``).

The invariance contract: with or without an extension, the fit and the devscore of every other
check are bit-identical, the base scenes are folded identically and the aggregate scores the same
clips. The aggregate's value is not invariant by construction: it reads every check's calibration,
the extended one included (here R5b turns from no_negatives into ok and joins the score). The error
profile never reads calibration rows (``diagnose stages`` runs on the base subset only).
"""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from calibration_cases import (
    CHECKS,
    config,
    extension_report,
    fit_meta,
    merged,
    pair_units,
    relabelled,
)
from calibration_cases import synthetic_report as base_report

from physverify.calibration.collect import level_data
from physverify.calibration.dev_folds import scene_folds
from physverify.calibration.dev_score import dev_score
from physverify.calibration.extension import ExtensionScope, extension_of, restrict
from physverify.calibration.fit import (
    FitInputs,
    LevelData,
    clip_scores,
    fit_calibration,
    fit_checks,
    fit_stratum,
)
from physverify.calibration.sigma_recalibration import SigmaPair
from physverify.core.errors import ConfigError
from physverify.datasets._base import ScopedSubset
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.synth.plan import family_of
from physverify.verification.sweep import CheckUnit

CONFIG = config(alpha=0.1, folds=3, bootstrap=30)
SCOPED = "R5b"
OTHERS = ("R1", "R3a")
R3B_UNIT = "dE_over_E"
KUBRIC = Path(__file__).parents[2] / "configs" / "datasets" / "kubric_synthetic.toml"


def arms() -> tuple[LevelData, LevelData, ExtensionScope]:
    base, extra = base_report(n_scenes=18), extension_report()
    scope = ExtensionScope({c.scene_id or "": frozenset({SCOPED}) for c in extra.clips})
    without = level_data(InputLevel.ORACLE, base, CHECKS, CONFIG, split=Split.CALIB)
    with_ext = level_data(
        InputLevel.ORACLE, merged(base, extra), CHECKS, CONFIG, split=Split.CALIB, scope=scope
    )
    return without, with_ext, scope


def dumped(model: object) -> str:
    return model.model_dump_json()  # type: ignore[attr-defined,no-any-return]


def test_extension_scenes_keep_only_their_checks() -> None:
    without, with_ext, scope = arms()
    added = [r for r in with_ext.rows if r.scene_id in scope.scenes]
    assert added and {r.check_id for r in added} == {SCOPED}
    assert [r for r in with_ext.rows if r.scene_id not in scope.scenes] == list(without.rows)
    extra_clips = {r.clip_id for r in added}
    assert {r.check_id for r in with_ext.results if r.clip_id in extra_clips} == {SCOPED}
    assert with_ext.extension_scenes == scope.scenes and not without.extension_scenes


def test_sigma_pairs_of_other_checks_are_dropped_on_extension_scenes() -> None:
    pairs = tuple(
        SigmaPair(scene, check, 0.1, 0.2, 0.05, *pair_units(check))
        for scene in ("s000", "x000")
        for check in ("R1", SCOPED)
    )
    data = LevelData(level=InputLevel.PERCEIVED, rows=(), results=(), sigma_pairs=pairs)
    kept = restrict(data, ExtensionScope({"x000": frozenset({SCOPED})}), {})
    expected = tuple(p for p in pairs if p.scene_id == "s000" or p.check_id == SCOPED)
    assert kept.sigma_pairs == expected
    assert restrict(data, ExtensionScope(), {}) is data


def test_the_other_checks_fit_bit_identically() -> None:
    without, with_ext, _ = arms()
    before, after = fit_checks(without, CHECKS, CONFIG), fit_checks(with_ext, CHECKS, CONFIG)
    assert all(dumped(before[c]) == dumped(after[c]) for c in OTHERS)
    assert before[SCOPED].status != "ok" and after[SCOPED].status == "ok"
    assert after["R1"].weight.value >= after[SCOPED].weight.value
    strata = [
        fit_calibration(FitInputs((data,), CHECKS, fit_meta()), CONFIG).strata[0]
        for data in (without, with_ext)
    ]
    assert all(dumped(strata[0].checks[c]) == dumped(strata[1].checks[c]) for c in OTHERS)


def test_the_aggregate_scores_the_same_clips() -> None:
    without, with_ext, _ = arms()
    model = fit_stratum(with_ext, CHECKS, "fast", CONFIG)
    assert clip_scores(model, with_ext) == clip_scores(model, without)
    plain = replace(with_ext, extension_scenes=frozenset())
    assert len(clip_scores(model, plain)) > len(clip_scores(model, without))


def test_base_scenes_are_folded_as_without_the_extension() -> None:
    without, with_ext, scope = arms()
    settings = CONFIG.calibration
    base, extended = scene_folds([without], settings), scene_folds([with_ext], settings)
    assert tuple(fold - scope.scenes for fold in extended) == base
    assert frozenset().union(*extended) & scope.scenes == scope.scenes


def test_the_other_checks_devscore_bit_identically() -> None:
    without, with_ext, _ = arms()
    both = (without, with_ext)
    reports = [dev_score((data,), CHECKS, fit_meta(), CONFIG, family_of).report for data in both]
    levels = [report.levels[0].checks for report in reports]
    assert all(dumped(levels[0][c]) == dumped(levels[1][c]) for c in OTHERS)
    assert dumped(levels[0][SCOPED]) != dumped(levels[1][SCOPED])


def test_an_extension_adds_only_the_scenes_the_base_lacks() -> None:
    base, extra = base_report(n_scenes=4), extension_report(n_scenes=3)
    stray = replace(extra.clips[0], clip_id="stray", scene_id=base.clips[0].scene_id)
    subset = ScopedSubset("extra", (SCOPED,), (*base.clips[:2], stray, *extra.clips))
    found = extension_of([subset, subset], base.clips, ("R1", SCOPED))
    assert found.clips == extra.clips
    assert found.scope.checks_by_scene == {c.scene_id: frozenset({SCOPED}) for c in extra.clips}
    assert found.label("full", base.clips) == "full"
    assert found.label("full", base.clips + extra.clips[:1]) == "full+extra"
    assert extension_of([], base.clips, ("R1",)).clips == ()


def test_an_extension_must_name_active_checks() -> None:
    base, extra = base_report(n_scenes=4), extension_report(n_scenes=3)
    with pytest.raises(ConfigError, match="checks_only=\\[\\]"):
        extension_of([ScopedSubset("extra", ("R9",), extra.clips)], base.clips, (SCOPED,))


def configured_scope() -> tuple[str, ...]:
    subsets = tomllib.loads(KUBRIC.read_text(encoding="utf-8"))["dataset"]["subsets"]
    return tuple(subsets["calib_f3_extra"]["checks_only"])


@pytest.mark.parametrize("level", [InputLevel.ORACLE, InputLevel.PERCEIVED])
def test_the_configured_r3b_extension_leaves_the_others_identical(level: InputLevel) -> None:
    (scoped,) = configured_scope()
    assert scoped == "R3b"
    checks = tuple(CheckUnit(scoped, R3B_UNIT) if c.check_id == SCOPED else c for c in CHECKS)
    base, extra = (
        relabelled(r, SCOPED, scoped, R3B_UNIT)
        for r in (base_report(n_scenes=18, level=level), extension_report(level=level))
    )
    scope = ExtensionScope({c.scene_id or "": frozenset({scoped}) for c in extra.clips})
    without = level_data(level, base, checks, CONFIG, split=Split.CALIB)
    both = merged(base, extra)
    with_ext = level_data(level, both, checks, CONFIG, split=Split.CALIB, scope=scope)
    assert {r.check_id for r in with_ext.rows if r.scene_id in scope.scenes} == {scoped}
    before, after = fit_checks(without, checks, CONFIG), fit_checks(with_ext, checks, CONFIG)
    assert all(dumped(before[c]) == dumped(after[c]) for c in OTHERS)
    assert (before[scoped].status, after[scoped].status) == ("no_negatives", "ok")
    model = fit_stratum(with_ext, checks, "fast", CONFIG)
    assert clip_scores(model, with_ext) == clip_scores(model, without)
    reports = [
        dev_score((d,), checks, fit_meta(), CONFIG, family_of).report for d in (without, with_ext)
    ]
    found = [report.levels[0].checks for report in reports]
    assert all(dumped(found[0][c]) == dumped(found[1][c]) for c in OTHERS)
    assert dumped(found[0][scoped]) != dumped(found[1][scoped])
