"""Sigma recalibration per check (D9) and per stage (D10), both cross-fitted by scene."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from physverify.calibration.schema import Ci, CoverageAt, SigmaCorrection, StageFactor
from physverify.calibration.settings import CalibrationSettings
from physverify.calibration.stats import (
    FloatArray,
    SceneBlock,
    grouped_folds,
    scene_bootstrap,
)
from physverify.diagnosis.cases import sample_stats
from physverify.diagnosis.error_profile_writer import (
    PROFILE_QUANTITIES,
    MetricSamples,
    StageSamples,
    entry_identity,
    finite_samples,
)
from physverify.diagnosis.report import nominal_coverage, z_coverage
from physverify.physics.uncertainty import MAD_TO_SIGMA
from physverify.schema.error_profile import ErrorEntry, ErrorProfile

COVERAGE_LEVELS = (1.0, 2.0)
BIAS_KEY = "bias"
FACTOR_KEY = "factor"
ORACLE_REASON = "oracle: exact inputs, sigma 0, not recalibrated"
FEW_PAIRS = "fewer than min_sigma_pairs (oracle, perceived) pairs: error not measured"
EntryKey = tuple[str, str, str]


@dataclass(frozen=True)
class SigmaPair:
    """One subject event seen by the oracle and by the perceived chain (``diagnosis.pairing``)."""

    scene_id: str
    check_id: str
    oracle: float
    perceived: float
    sigma: float

    @property
    def delta(self) -> float:
        return self.perceived - self.oracle


def coverage(z: FloatArray | Sequence[float]) -> tuple[CoverageAt, ...]:
    return tuple(
        CoverageAt(
            k=c.k,
            nominal=c.nominal,
            observed=c.observed if math.isfinite(c.observed) else None,
            n=c.n,
        )
        for c in z_coverage(np.asarray(z, dtype=np.float64), COVERAGE_LEVELS)
    )


def oracle_correction() -> SigmaCorrection:
    return SigmaCorrection(
        bias=0.0,
        bias_ci=Ci(),
        factor=1.0,
        factor_ci=Ci(),
        n_pairs=0,
        coverage_before=(),
        coverage_after=(),
        reason=ORACLE_REASON,
        estimated=False,
    )


def usable(pairs: Sequence[SigmaPair]) -> tuple[SigmaPair, ...]:
    return tuple(
        p for p in pairs if math.isfinite(p.delta) and math.isfinite(p.sigma) and p.sigma > 0.0
    )


def bias_factor(pairs: Sequence[SigmaPair]) -> tuple[float, float]:
    """b = median(m_P - m_O), k = median(|z|) / 0.6745 with z = (Delta - b) / sigma_P."""
    delta = np.asarray([p.delta for p in pairs])
    sigma = np.asarray([p.sigma for p in pairs])
    bias = float(np.median(delta))
    return bias, float(np.median(np.abs(delta - bias) / sigma)) / MAD_TO_SIGMA


def fit_sigma(pairs: Sequence[SigmaPair], settings: CalibrationSettings) -> SigmaCorrection:
    found = usable(pairs)
    before = coverage([p.delta / p.sigma for p in found])
    if len(found) < settings.min_sigma_pairs:
        return SigmaCorrection(
            bias=0.0,
            bias_ci=Ci(),
            factor=None,
            factor_ci=Ci(),
            n_pairs=len(found),
            coverage_before=before,
            coverage_after=(),
            reason=FEW_PAIRS,
        )
    bias, factor = bias_factor(found)
    blocks = [SceneBlock(p.scene_id, p) for p in found]
    ci = scene_bootstrap(blocks, _bias_factor_map, settings.bootstrap, settings.seed, settings.ci)
    return SigmaCorrection(
        bias=bias,
        bias_ci=ci.get(BIAS_KEY, Ci()),
        factor=factor,
        factor_ci=ci.get(FACTOR_KEY, Ci()),
        n_pairs=len(found),
        coverage_before=before,
        coverage_after=coverage(cross_fitted_z(found, settings)),
    )


def _bias_factor_map(pairs: Sequence[SigmaPair]) -> dict[str, float]:
    bias, factor = bias_factor(pairs)
    return {BIAS_KEY: bias, FACTOR_KEY: factor}


def cross_fitted_z(pairs: Sequence[SigmaPair], settings: CalibrationSettings) -> FloatArray:
    """(Delta - b) / (k sigma) with b and k fitted without the pair's fold of scenes."""
    z: list[float] = []
    for fold in grouped_folds([p.scene_id for p in pairs], settings.folds, settings.seed):
        train = [p for p in pairs if p.scene_id not in fold]
        if not train:
            continue
        bias, factor = bias_factor(train)
        if factor <= 0.0:
            continue
        z += [(p.delta - bias) / (factor * p.sigma) for p in pairs if p.scene_id in fold]
    return np.asarray(z, dtype=np.float64)


# --- D10: per stage -----------------------------------------------------------------------------


def _clip_residuals(samples: MetricSamples, quantity: str) -> dict[str, FloatArray]:
    return {
        clip_id: finite_samples((outcome.samples[quantity],))
        for clip_id, outcome in samples.outcomes.items()
        if quantity in outcome.samples
    }


def oof_z(
    residuals: Mapping[str, FloatArray], scenes: Mapping[str, str], settings: CalibrationSettings
) -> tuple[FloatArray, FloatArray]:
    """Out-of-fold z = (r - b) / sigma (b, sigma from the other folds), and the fold of each z."""
    z: list[FloatArray] = []
    fold_of: list[FloatArray] = []
    folds = grouped_folds([scenes.get(c, c) for c in residuals], settings.folds, settings.seed)
    for index, fold in enumerate(folds):
        held = [c for c in residuals if scenes.get(c, c) in fold]
        train = finite_samples(r for c, r in residuals.items() if c not in held)
        bias, sigma, _, _ = sample_stats(train)
        if not math.isfinite(sigma) or sigma <= 0.0:
            continue
        values = finite_samples(residuals[c] for c in held)
        z.append((values - bias) / sigma)
        fold_of.append(np.full(values.size, index, dtype=np.float64))
    if not z:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(z), np.concatenate(fold_of)


def stage_factor(z: FloatArray) -> float | None:
    """f = q_0.683(|z|): the factor that gives |z| / f <= 1 its Gaussian coverage."""
    if not z.size:
        return None
    return float(np.quantile(np.abs(z), nominal_coverage(1.0)))


def _cross_fitted_after(z: FloatArray, folds: FloatArray) -> FloatArray:
    scaled: list[FloatArray] = []
    for index in np.unique(folds):
        factor = stage_factor(z[folds != index])
        if factor is not None and factor > 0.0:
            scaled.append(z[folds == index] / factor)
    return np.concatenate(scaled) if scaled else np.zeros(0)


def stage_factors(
    samples: StageSamples, scenes: Mapping[str, str], settings: CalibrationSettings
) -> tuple[StageFactor, ...]:
    found: list[StageFactor] = []
    for measured in samples:
        for quantity in PROFILE_QUANTITIES.get(measured.metric, ()):
            z, folds = oof_z(_clip_residuals(measured, quantity), scenes, settings)
            found.append(
                StageFactor(
                    stage=measured.provider.stage.value,
                    provider=measured.provider.name,
                    quantity=quantity,
                    factor=stage_factor(z),
                    n=int(z.size),
                    coverage_before=coverage(z),
                    coverage_after=coverage(_cross_fitted_after(z, folds)),
                )
            )
    return tuple(found)


def recalibrate_entry(entry: ErrorEntry, factor: float) -> ErrorEntry:
    """r' = b + f (r - b), sigma' = f sigma: the Monte Carlo resamples the residuals."""
    residuals = tuple(entry.bias + factor * (r - entry.bias) for r in entry.residuals)
    return dataclasses.replace(entry, sigma=factor * entry.sigma, residuals=residuals)


def recalibrate_profile(profile: ErrorProfile, factors: Mapping[EntryKey, float]) -> ErrorProfile:
    """Entries with a factor are rescaled about their bias; the others are kept unchanged."""
    entries = tuple(
        recalibrate_entry(e, factors[entry_identity(e)]) if entry_identity(e) in factors else e
        for e in profile.entries
    )
    return dataclasses.replace(profile, entries=entries)


def factor_map(factors: Sequence[StageFactor]) -> dict[EntryKey, float]:
    return {
        (f.stage, f.provider, f.quantity): f.factor
        for f in factors
        if f.factor is not None and math.isfinite(f.factor) and f.factor > 0.0
    }
