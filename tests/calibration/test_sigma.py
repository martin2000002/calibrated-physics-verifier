"""Sigma recalibration per check (D9, AC8) and per stage (D10)."""

from __future__ import annotations

import numpy as np
import pytest
from calibration_cases import check_calibration, settings, view

from physverify.calibration.sigma import (
    FEW_PAIRS,
    SigmaPair,
    factor_map,
    fit_sigma,
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
