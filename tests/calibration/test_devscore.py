"""``calibration.dev_score``: the development metric of the campaign."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from calibration_cases import (
    CHECKS,
    LEVELS,
    UNITS,
    _rows,
    check_calibration,
    clip,
    config,
    fit_meta,
    record,
)

from physverify.calibration.artifact import Reading
from physverify.calibration.check_weights import Flag, raw_weight
from physverify.calibration.collect import level_data
from physverify.calibration.dev_folds import fold_parts, scene_folds, split_data
from physverify.calibration.dev_score import (
    DEVSCORE_SCHEMA,
    STRATIFIED_NOTE,
    DevScore,
    LevelDevScore,
    oof_flags,
    require_bootstrap,
    selection,
)
from physverify.calibration.dev_score import dev_score as compute
from physverify.calibration.fit import FitInputs, LevelData, fit_calibration, fit_checks
from physverify.calibration.oof import DevScoreOof
from physverify.calibration.sigma_recalibration import SigmaPair
from physverify.calibration.stats import auroc
from physverify.calibration.table import Role
from physverify.calibration.thresholds import fit_tau
from physverify.core.errors import UsageError
from physverify.schema.checks import CheckResult, InputLevel
from physverify.schema.clip import Split
from physverify.synth.plan import family_of, scene_id_of
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import VerificationReport
from physverify.verification.thresholding import ThresholdParams

SCENES = 18
CONFIG = config(alpha=0.1, folds=3, bootstrap=30)
SMALL = config(alpha=0.1, folds=2, bootstrap=20)
"""alpha 0.1: nine clean scenes reach the conformal rank, so small folds keep a tau."""
VOLATILE = ("created_utc", "run_id")


def plain(index: int) -> str:
    return f"s{index:03d}"


def two_families(index: int) -> str:
    return scene_id_of("fa" if index % 2 else "fb", index)


def report(
    n_scenes: int = SCENES, seed: int = 0, scene_name: Callable[[int], str] = plain
) -> VerificationReport:
    """Per scene: a clean clip, R1 at three levels (mu = 0.05 * 2^(L-1) over clean noise 0.01)
    and R3a 'perturbed' at three levels with the clean noise only: R1 separable and monotone,
    R3a random, R5b without a clean event."""
    rng = np.random.default_rng(seed)
    table, results, clips = [], [], []
    for s in range(n_scenes):
        scene = scene_name(s)
        found = [(clip(f"{scene}__none", scene), None, 0.0)]
        for name, target in (("r1_penetration", "R1"), ("r3a_noise", "R3a")):
            for lv in LEVELS:
                mu = 0.05 * 2 ** (lv - 1)
                pert = record(name, (target,), mu, UNITS[target], lv)
                found.append((clip(f"{scene}__{name}_L{lv}", scene, perturbation=pert), target, mu))
        for manifest, target, mu in found:
            magnitudes: dict[str, float | None] = {
                "R1": abs(float(rng.normal(0.0, 0.01))),
                "R3a": abs(float(rng.normal(0.0, 0.02))),
                "R5b": None,
            }
            if target == "R1":
                magnitudes["R1"] = mu + float(rng.normal(0.0, 0.005))
            rows, found_results = _rows(manifest, magnitudes, 0.0, InputLevel.ORACLE)
            table += rows
            results += found_results
            clips.append(manifest)
    return VerificationReport(results=tuple(results), table=tuple(table), clips=tuple(clips))


REPORT = report()


def levels(source: VerificationReport = REPORT) -> tuple[LevelData, LevelData]:
    """The oracle, and a perceived level without sigma recalibration (``--no-sigma``)."""
    oracle = level_data(InputLevel.ORACLE, source, CHECKS, CONFIG, split=Split.CALIB)
    perceived = level_data(
        InputLevel.PERCEIVED, source, CHECKS, CONFIG, split=Split.CALIB, pairs=None
    )
    return oracle, perceived


@pytest.fixture(scope="module")
def dev() -> DevScore:
    return compute(levels(), CHECKS, fit_meta(), CONFIG, family_of)


def stable(text: str) -> dict[str, object]:
    return {k: v for k, v in json.loads(text).items() if k not in VOLATILE}


def test_two_runs_differ_only_in_time_and_run_id() -> None:
    """On the oracle level of a smaller case, to stay fast."""
    small = levels(report(20, seed=3))[:1]
    first = compute(small, CHECKS, fit_meta(run_id="a"), SMALL, family_of)
    later = fit_meta(run_id="b", created_utc="2026-10-21T00:00:00Z")
    second = compute(small, CHECKS, later, SMALL, family_of)
    assert stable(first.report.to_json()) == stable(second.report.to_json())
    assert json.loads(first.report.to_json())["schema"] == DEVSCORE_SCHEMA


def test_monotonicity_and_the_in_sample_weight_are_the_fits(dev: DevScore) -> None:
    """The full fit inside devscore is ``calibrate fit``'s artifact."""
    fitted = fit_calibration(FitInputs(levels(), CHECKS, fit_meta()), CONFIG)
    assert dev.artifact.to_json() == fitted.to_json()
    raw = fit_checks(levels()[0], CHECKS, CONFIG)
    (oracle_fit, _), oracle = fitted.strata, dev.report.levels[0]
    for check_id, score in oracle.checks.items():
        assert score.h1 == oracle_fit.checks[check_id].monotonicity
        assert score.tau == oracle_fit.checks[check_id].tau
        assert score.j_in_sample == raw[check_id].weight
        assert score.meets_h1 is score.h1.h1_pass


def test_the_folds_partition_the_scenes_and_refit_tau(dev: DevScore) -> None:
    """Disjoint folds covering every calib scene; tau_k is fit_tau without fold k."""
    oracle = levels()[0]
    folds = scene_folds([oracle], CONFIG.calibration)
    scenes = {r.scene_id for r in oracle.rows}
    assert sum(len(f) for f in folds) == len(scenes) == len(frozenset().union(*folds))
    assert [f.n_scenes for f in dev.report.folds] == [len(f) for f in folds]
    clean = [r for r in oracle.rows if r.check_id == "R1" and r.role is Role.CLEAN]
    expected = tuple(
        fit_tau(
            [r for r in clean if r.scene_id not in fold], 0.0, CONFIG.calibration.alpha, 0.0
        ).value
        for fold in folds
    )
    assert dev.report.levels[0].checks["R1"].tau_by_fold == expected


def _pairs(data: LevelData) -> tuple[SigmaPair, ...]:
    rng = np.random.default_rng(5)
    scenes = sorted({r.scene_id for r in data.rows})
    return tuple(
        SigmaPair(scene, "R1", 0.01, 0.01 + float(rng.normal(0.0, 0.002)), 0.002)
        for scene in scenes
        for _ in range(2)
    )


def _corrupt(data: LevelData, fold: frozenset[str]) -> LevelData:
    """Every row, result and sigma pair of ``fold`` changed (and one result dropped)."""
    held = {r.clip_id for r in data.rows if r.scene_id in fold}

    def big(result: CheckResult) -> CheckResult:
        return replace(result, magnitude=1e6)

    rows = tuple(
        replace(r, magnitude=1e6, subjects=tuple(big(s) for s in r.subjects))
        if r.scene_id in fold
        else r
        for r in data.rows
    )
    results = tuple(big(r) if r.clip_id in held else r for r in data.results)
    first = next(i for i, r in enumerate(results) if r.clip_id in held)
    pairs = tuple(
        replace(p, perceived=5.0) if p.scene_id in fold else p for p in data.sigma_pairs or ()
    )
    return LevelData(data.level, rows, results[:first] + results[first + 1 :], pairs)


def test_nothing_of_the_held_out_fold_reaches_its_stratum() -> None:
    """The fold's stratum and tau are unchanged by anything inside the fold."""
    base = levels()[1]
    data = LevelData(base.level, base.rows, base.results, _pairs(base))
    fold, other = scene_folds([data], CONFIG.calibration)[:2]
    corrupt = _corrupt(data, fold)
    before, control = fold_parts(data, CHECKS, "fast", CONFIG, [fold, other])
    after, leaked = fold_parts(corrupt, CHECKS, "fast", CONFIG, [fold, other])
    assert before.stratum.checks["R1"].sigma.n_pairs > 0
    assert after.stratum.to_json() == before.stratum.to_json()
    assert leaked.stratum.to_json() != control.stratum.to_json()  # the corruption is visible


def test_split_keeps_no_sigma_and_drops_results_without_a_row() -> None:
    """--no-sigma stays None in every fold; a result of a clip without a row is dropped."""
    data = levels()[1]
    orphan = replace(data.results[0], clip_id="no-row")
    data = LevelData(data.level, data.rows, (*data.results, orphan), None)
    fold = scene_folds([data], CONFIG.calibration)[0]
    train, held = split_data(data, fold)
    assert train.sigma_pairs is None and held.sigma_pairs is None
    assert len(train.results) + len(held.results) == len(data.results) - 1
    assert {r.scene_id for r in held.rows} == fold


def test_a_fold_without_tau_flags_nothing() -> None:
    """tau_k None leaves the fold's rows as not FAIL."""
    rows = [r for r in levels()[0].rows if r.check_id == "R1"]
    policy = ThresholdPolicy(ThresholdParams())
    flags = oof_flags(rows, check_calibration(tau=None), policy)
    assert flags and not any(f.flag for f in flags)
    assert {f.positive for f in flags} == {True, False}
    assert any(f.flag for f in oof_flags(rows, check_calibration(tau=0.03), policy))


def test_the_selection_counts_the_separable_monotone_check(dev: DevScore) -> None:
    """R1 is monotone and J_cv^lo > 0, the random R3a and the eventless R5b do not."""
    _oracle, perceived = dev.report.levels
    assert perceived.n_met == 1 and perceived.met == ("R1",)
    assert not perceived.checks["R3a"].meets_both
    assert dev.report.selection is not None
    assert dev.report.selection.oracle_evaluable == ("R1",)
    assert dev.report.selection.all_oracle_evaluable_met is True
    assert dev.report.selection.tiebreak_auroc_cv == perceived.auroc_cv.value


def _met(level: LevelDevScore, *met: str) -> LevelDevScore:
    return level.model_copy(update={"met": met, "n_met": len(met)})


def test_the_early_freeze_needs_every_oracle_evaluable_check(dev: DevScore) -> None:
    """The other way round, and the edge cases: no oracle level, nothing evaluable."""
    oracle, perceived = dev.report.levels
    wider = selection([_met(oracle, "R1", "R3a"), perceived])
    assert wider is not None and wider.all_oracle_evaluable_met is False
    nothing = selection([_met(oracle), perceived])
    assert nothing is not None and nothing.all_oracle_evaluable_met is False
    alone = selection([perceived])
    assert alone is not None and alone.oracle_evaluable is None
    assert alone.all_oracle_evaluable_met is None
    assert selection([oracle]) is None


def test_the_score_auroc_is_out_of_fold(dev: DevScore) -> None:
    """Marked out of fold; the separable R1 clips rank perfectly once R3a weighs nothing."""
    for level in dev.report.levels:
        assert level.auroc_cv.in_sample is False
        assert level.auroc_cv.n_negative == SCENES


def test_perfect_separation_out_of_fold_is_one() -> None:
    """A report with R1 perturbations only (every perturbed clip separable) gives 1.0."""
    only_r1 = report(20, seed=1)
    kept = [c for c in only_r1.clips if "r3a_noise" not in c.clip_id]
    ids = {c.clip_id for c in kept}
    narrow = VerificationReport(
        results=tuple(r for r in only_r1.results if r.clip_id in ids),
        table=tuple(r for r in only_r1.table if r.clip_id in ids),
        clips=tuple(kept),
    )
    result = compute(levels(narrow)[:1], CHECKS, fit_meta(), SMALL, family_of)
    assert result.report.levels[0].auroc_cv.value == 1.0


def test_bootstrap_zero_is_refused() -> None:
    """Never bootstrap 0 (every weight would be 0)."""
    with pytest.raises(UsageError, match="bootstrap"):
        require_bootstrap(config(bootstrap=0).calibration)
    with pytest.raises(UsageError, match="bootstrap"):
        compute(levels()[:1], CHECKS, fit_meta(), config(bootstrap=0), family_of)


def test_folds_without_a_sigma_factor_are_counted() -> None:
    """A fold whose training lacks min_sigma_pairs informative pairs has no factor
    (its subjects abstain); the count is reported per check."""
    oracle, base = levels()
    data = LevelData(base.level, base.rows, base.results, _pairs(base))
    found = compute((oracle, data), CHECKS, fit_meta(), CONFIG, family_of).report
    perceived = next(level for level in found.levels if level.level is InputLevel.PERCEIVED)
    assert perceived.checks["R1"].folds_without_sigma_factor == 0
    assert perceived.checks["R3a"].folds_without_sigma_factor == CONFIG.calibration.folds
    zeros = tuple(replace(p, oracle=0.0, perceived=0.0) for p in data.sigma_pairs or ())
    censored = LevelData(base.level, base.rows, base.results, zeros)
    blind = compute((oracle, censored), CHECKS, fit_meta(), CONFIG, family_of).report
    level = next(item for item in blind.levels if item.level is InputLevel.PERCEIVED)
    assert level.checks["R1"].folds_without_sigma_factor == CONFIG.calibration.folds
    oracle_level = next(item for item in blind.levels if item.level is InputLevel.ORACLE)
    assert all(c.folds_without_sigma_factor == 0 for c in oracle_level.checks.values())


def test_scenes_without_a_family_keep_the_global_clause(dev: DevScore) -> None:
    """No family: nothing stratified, the clause is monotonicity and J_cv^lo > 0 as in 1.1.0."""
    for level in dev.report.levels:
        assert level.stratified is None and level.stratification is not None
        assert not level.stratification.applies
        assert (level.n_met, level.met) == (level.n_met_global, level.met_global)
        for check in level.checks.values():
            assert check.stratified is None and check.meets_stratified is None
            assert check.meets_clause == check.meets_both
    chosen = dev.report.selection
    assert chosen is not None and chosen.tiebreak_auroc_cv == chosen.tiebreak_auroc_cv_global
    assert STRATIFIED_NOTE not in dev.report.scope_note


def test_family_scenes_add_the_stratified_clause() -> None:
    """Two families on every scene: each check and the score carry both readings, and a check
    counts only when its stratified bounds pass too."""
    source = report(SCENES, scene_name=two_families)
    found = compute(levels(source)[:1], CHECKS, fit_meta(), CONFIG, family_of).report
    (level,) = found.levels
    assert level.stratification is not None and level.stratification.families == ("fa", "fb")
    assert level.stratified is not None and set(level.stratified.auroc_cv) == set(Reading)
    for check in level.checks.values():
        expected = check.meets_both and check.meets_stratified is not False
        assert check.meets_clause == expected
    assert level.met == tuple(sorted(c for c, s in level.checks.items() if s.meets_clause))
    assert level.checks["R1"].meets_clause
    r1 = level.checks["R1"].stratified
    assert r1 is not None and r1.j_cv is not None and r1.j_cv.informedness > 0.9
    assert found.scope_note.endswith(STRATIFIED_NOTE)


def test_the_out_of_fold_values_give_back_the_score_auroc(dev: DevScore) -> None:
    """``devscore_oof.json`` holds the clip scores AUROC_cv was computed from."""
    assert DevScoreOof.model_validate_json(dev.oof.to_json()) == dev.oof
    for level in dev.report.levels:
        values = dev.oof.level(level.level)
        assert values is not None
        scored = [(c.value, c.perturbed) for c in values.aggregate if c.value is not None]
        pos = [v for v, positive in scored if positive]
        neg = [v for v, positive in scored if not positive]
        assert auroc(pos, neg) == level.auroc_cv.value
        for check_id, check in level.checks.items():
            judged = [v for v in values.checks[check_id] if v.flag is not None]
            flags = [Flag(v.scene_id, v.role is Role.POSITIVE, v.flag) for v in judged]
            assert raw_weight(flags, CONFIG.calibration) == check.j_cv
