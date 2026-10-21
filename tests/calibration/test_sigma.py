"""Sigma recalibration per check (D9, AC8) and per stage (D10)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from calibration_cases import check_calibration, settings, view

from physverify.calibration.sigma import (
    FEW_PAIRS,
    SigmaPair,
    bias_factor,
    cross_fitted_z,
    factor_map,
    fit_sigma,
    informative,
    oof_z,
    oracle_correction,
    recalibrate_profile,
    stage_factor,
)
from physverify.diagnosis.report import profile_hash
from physverify.schema.error_profile import ErrorEntry, ErrorProfile
from physverify.schema.stages import Stage

SCALE = 2.0
BIAS = 0.03


def pairs(rng: np.random.Generator, n: int = 200, scale: float = SCALE) -> list[SigmaPair]:
    """Delta = bias + scale * sigma * N(0, 1): the propagated sigma is too small by ``scale``."""
    sigma = rng.uniform(0.01, 0.05, n)
    delta = BIAS + scale * sigma * rng.normal(0.0, 1.0, n)
    return [
        SigmaPair(f"s{i % 40}", "R1", 0.1, 0.1 + float(d), float(s))
        for i, (d, s) in enumerate(zip(delta, sigma, strict=True))
    ]


def test_the_factor_and_bias_are_recovered_and_coverage_is_restored() -> None:
    """AC8: k within 10 % of the true scale; |z| / k <= 1 on a fresh sample covers 68.3 % +- 5 pp.
    The bias is a median of 200 deltas with sd up to 0.1 (standard error about 0.005)."""
    rng = np.random.default_rng(0)
    found = fit_sigma(pairs(rng), settings())
    assert found.factor is not None
    assert 1.8 <= found.factor <= 2.2
    assert found.bias == pytest.approx(BIAS, abs=0.02)
    assert found.bias_ci.low is not None and found.bias_ci.high is not None
    assert found.bias_ci.low < found.bias < found.bias_ci.high
    fresh = pairs(rng, 2000)
    z = [(p.delta - found.bias) / (found.factor * p.sigma) for p in fresh]
    assert float(np.mean(np.abs(z) <= 1.0)) == pytest.approx(0.683, abs=0.05)
    after = next(c for c in found.coverage_after if c.k == 1.0)
    assert after.observed == pytest.approx(0.683, abs=0.05)
    before = next(c for c in found.coverage_before if c.k == 1.0)
    assert before.observed is not None and before.observed < 0.5


def test_the_factor_is_stable_over_replicates() -> None:
    rng = np.random.default_rng(1)
    factors = [fit_sigma(pairs(rng), settings(bootstrap=0)).factor for _ in range(25)]
    assert float(np.median([f for f in factors if f is not None])) == pytest.approx(SCALE, rel=0.03)


def test_too_few_pairs_leave_the_error_unmeasured() -> None:
    found = fit_sigma(pairs(np.random.default_rng(0), 5), settings())
    assert (found.factor, found.reason, found.n_pairs) == (None, FEW_PAIRS, 5)
    stratum = view({"R1": check_calibration(tau=0.1, factor=None)})
    bias, factor = stratum.sigma_correction("R1")
    assert bias == 0.0 and np.isnan(factor)
    assert oracle_correction().factor == 1.0 and not oracle_correction().estimated


def test_stage_factor_of_scaled_out_of_fold_z() -> None:
    rng = np.random.default_rng(2)
    z = SCALE * rng.normal(0.0, 1.0, 4000)
    assert stage_factor(z) == pytest.approx(SCALE, rel=0.05)
    residuals = {f"c{i}": rng.normal(0.5, 1.0, 50) for i in range(20)}
    oof, folds = oof_z(residuals, {c: c for c in residuals}, settings())
    assert oof.size == 1000 and folds.size == 1000
    assert stage_factor(oof) == pytest.approx(1.0, rel=0.1)


def entry(quantity: str, residuals: tuple[float, ...]) -> ErrorEntry:
    return ErrorEntry(
        stage=Stage.DEPTH,
        provider="fake",
        provider_version="1",
        quantity=quantity,
        unit="",
        bias=0.1,
        sigma=0.2,
        n=len(residuals),
        residuals=residuals,
    )


def test_recalibrate_profile_keeps_bias_and_scales_sigma_and_residuals() -> None:
    profile = ErrorProfile(
        created_utc="t",
        run_id="r",
        commit="",
        entries=(entry("abs_rel", (0.0, 0.2, 0.4)), entry("scale_rel", (0.1,))),
    )
    factors = factor_map([])
    assert recalibrate_profile(profile, factors) == profile
    out = recalibrate_profile(profile, {("depth", "fake", "abs_rel"): 2.0})
    scaled, kept = out.entries
    assert scaled.bias == 0.1 and scaled.sigma == pytest.approx(0.4)
    assert scaled.residuals == pytest.approx((-0.1, 0.3, 0.7))
    assert kept == profile.entries[1]
    assert profile_hash(out) != profile_hash(profile)


# --- it1 change C: the D9 factor over the informative pairs (T12-T15, J3) ----------------------


def censored_pairs(
    rng: np.random.Generator, n: int = 400, zero_share: float = 0.7, scale: float = SCALE
) -> list[SigmaPair]:
    """``zero_share`` of 0-against-0 pairs (a magnitude clipped at 0 on both sides); the rest
    have Delta = scale * sigma * N(0, 1) about a nonzero oracle."""
    found: list[SigmaPair] = []
    for i in range(n):
        sigma = float(rng.uniform(0.01, 0.05))
        if i < zero_share * n:
            found.append(SigmaPair(f"s{i % 40}", "R1", 0.0, 0.0, sigma))
        else:
            delta = scale * sigma * float(rng.normal(0.0, 1.0))
            found.append(SigmaPair(f"s{i % 40}", "R1", 0.1, 0.1 + delta, sigma))
    return found


def test_t12_zero_pairs_collapse_the_old_factor_and_not_the_new_one() -> None:
    found = censored_pairs(np.random.default_rng(4))
    old = fit_sigma(found, settings(sigma_pair_filter="all"))
    new = fit_sigma(found, settings(sigma_pair_filter="nonzero"))
    assert old.factor == 0.0
    assert new.factor == pytest.approx(SCALE, rel=0.2)
    assert new.factor_ci.low is not None and new.factor_ci.low > 0.0
    assert new.n_informative == len(informative(found)) == 120
    assert new.coverage_after_informative and new.coverage_after_degenerate
    degenerate = next(c for c in new.coverage_after_degenerate if c.k == 1.0)
    assert degenerate.n == 280


def test_t13_few_informative_pairs_leave_the_error_unmeasured() -> None:
    found = censored_pairs(np.random.default_rng(5), n=100, zero_share=0.95)
    result = fit_sigma(found, settings(sigma_pair_filter="nonzero"))
    assert (result.factor, result.reason, result.n_pairs) == (None, FEW_PAIRS, 100)
    assert result.n_informative == 5
    assert fit_sigma(found, settings(sigma_pair_filter="all")).factor == 0.0


def _old_bias_factor(found: list[SigmaPair]) -> tuple[float, float]:
    """D9 as it was before 2026-10-20 (the reference for T14)."""
    delta = np.asarray([p.delta for p in found])
    sigma = np.asarray([p.sigma for p in found])
    bias = float(np.median(delta))
    return bias, float(np.median(np.abs(delta - bias) / sigma)) / 0.6745


def test_t14_the_all_filter_is_the_old_estimator() -> None:
    found = censored_pairs(np.random.default_rng(6), zero_share=0.3)
    assert bias_factor(found, "all") == _old_bias_factor(found)
    result = fit_sigma(found, settings(sigma_pair_filter="all"))
    assert (result.bias, result.factor) == _old_bias_factor(found)
    assert result.n_informative is None
    assert result.coverage_after_informative == () and result.coverage_after_degenerate == ()


@pytest.mark.parametrize("zero_share", [0.0, 0.3, 0.7])
def test_t15_the_bias_does_not_depend_on_the_filter(zero_share: float) -> None:
    found = censored_pairs(np.random.default_rng(7), zero_share=zero_share)
    old = fit_sigma(found, settings(sigma_pair_filter="all"))
    new = fit_sigma(found, settings(sigma_pair_filter="nonzero"))
    assert new.bias == old.bias
    assert new.bias_ci == old.bias_ci


def test_a_replica_with_fewer_than_two_informative_pairs_has_no_factor() -> None:
    zeros = [SigmaPair(f"s{i}", "R1", 0.0, 0.0, 0.01) for i in range(10)]
    one = [*zeros, SigmaPair("s9", "R1", 0.1, 0.2, 0.01)]
    assert math.isnan(bias_factor(one, "nonzero")[1])
    assert bias_factor(one, "all")[1] == 0.0
    rng = np.random.default_rng(8)
    found = [SigmaPair(f"s{i}", "R1", 0.0, 0.0, 0.01) for i in range(1, 31)]
    found += [
        SigmaPair("s0", "R1", 0.1, 0.1 + float(rng.normal(0.0, 0.02)), 0.01) for _ in range(10)
    ]
    result = fit_sigma(found, settings(sigma_pair_filter="nonzero"))
    assert result.factor is not None
    assert result.factor_ci.non_finite > 0


def test_cross_fitting_skips_a_fold_whose_training_lacks_informative_pairs() -> None:
    """Every informative pair sits in scene s0: the fold holding s0 trains without any and is
    skipped, the others are scored with the filter."""
    found = [SigmaPair(f"s{i % 6}", "R1", 0.0, 0.0, 0.01) for i in range(30)]
    found += [SigmaPair("s0", "R1", 0.1, 0.1 + 0.01 * k, 0.01) for k in range(-3, 4)]
    cfg = settings(sigma_pair_filter="nonzero", folds=3)
    z = cross_fitted_z(found, cfg)
    held_with_s0 = [p for p in found if p.scene_id in {"s0"}]
    assert z.shape[1] == 2
    assert z.shape[0] < len(found)
    assert z.shape[0] <= len(found) - len(held_with_s0)
    assert not bool(z[:, 1].any())
