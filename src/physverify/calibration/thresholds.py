"""Decision threshold L_C (D5, Juicio 1), honest FPR (D6), TPR per level and L_D (D7, Juicio 3)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from physverify.calibration.schema import Ci, DetectionLimit, Fpr, LevelTpr, OpenSide, Rate
from physverify.calibration.settings import CalibrationSettings
from physverify.calibration.stats import (
    clustered_rate,
    conformal_quantile,
    fit_logistic,
    grouped_folds,
    median_or_none,
    profile_interval,
    rate,
)
from physverify.calibration.table import (
    CalibrationRow,
    Role,
    corrected,
    require_one_clean_per_scene,
)
from physverify.schema.checks import Verdict
from physverify.verification.policies.threshold import ThresholdPolicy
from physverify.verification.runner import ThresholdEntry

DECIDED = (Verdict.PASS, Verdict.FAIL)


@dataclass(frozen=True)
class Rule:
    """The verdict rule of ``ThresholdPolicy`` with the measured bias and sigma factor."""

    policy: ThresholdPolicy
    bias: float
    factor: float | None

    def verdict(self, row: CalibrationRow, tau: float | None) -> Verdict:
        if tau is None or row.magnitude is None or not row.scored:
            return Verdict.ABSTAIN
        entry = ThresholdEntry(tau=tau, bias=self.bias, sigma_factor=self.factor)
        return self.policy.judge(row.magnitude, row.sigma, entry)[0]

    def fails(self, row: CalibrationRow, tau: float | None) -> bool:
        return self.verdict(row, tau) is Verdict.FAIL


@dataclass(frozen=True)
class Tau:
    """``value`` = max(q, tolerance); ``conformal`` = q."""

    value: float | None
    sufficient: bool
    n: int
    conformal: float | None = None


def fit_tau(clean: Sequence[CalibrationRow], bias: float, alpha: float, tolerance: float) -> Tau:
    """Conformal quantile q over one clean clip per scene; tau = max(q, tolerance). The tolerance
    keeps a q made of floating-point noise (oracle clean ~1e-14) from becoming the threshold."""
    require_one_clean_per_scene(clean)
    values = [corrected(row, bias) for row in clean if row.scored]
    found = conformal_quantile(values, alpha)
    value = None if found.value is None else max(found.value, tolerance)
    return Tau(value, found.sufficient, len(values), found.value)


def cv_verdicts(
    clean: Sequence[CalibrationRow], rule: Rule, settings: CalibrationSettings, tolerance: float
) -> dict[str, Verdict]:
    """Per scored clean clip: its verdict against a tau fitted without its fold of scenes."""
    scored = [row for row in clean if row.scored]
    found: dict[str, Verdict] = {}
    for fold in grouped_folds([r.scene_id for r in scored], settings.folds, settings.seed):
        train = [r for r in scored if r.scene_id not in fold]
        tau = fit_tau(train, rule.bias, settings.alpha, tolerance).value
        for row in (r for r in scored if r.scene_id in fold):
            found[row.clip_id] = rule.verdict(row, tau)
    return found


def false_positive_rates(verdicts: Sequence[Verdict], ci: float) -> tuple[Rate, Rate]:
    """(FAIL over all, FAIL over decided) of one clean clip per scene."""
    fails = sum(v is Verdict.FAIL for v in verdicts)
    decided = sum(v in DECIDED for v in verdicts)
    return rate(fails, len(verdicts), ci), rate(fails, decided, ci)


def fpr_summary(
    rows: Sequence[CalibrationRow],
    rule: Rule,
    tau: float | None,
    cv: Mapping[str, Verdict],
    settings: CalibrationSettings,
) -> Fpr:
    clean = [r for r in rows if r.role is Role.CLEAN and r.scored]
    off = [r for r in rows if r.role is Role.OFF_TARGET and r.scored]
    off_rate = off_target_rate(off, rule, tau, settings)
    off_fails = [r for r in off if rule.fails(r, tau)]
    cv_total, cv_decided = false_positive_rates(list(cv.values()), settings.ci)
    in_total, in_decided = false_positive_rates([rule.verdict(r, tau) for r in clean], settings.ci)
    low = off_rate.ci.low
    return Fpr(
        cv=cv_total,
        cv_decided=cv_decided,
        in_sample=in_total,
        in_sample_decided=in_decided,
        off_target=off_rate,
        undeclared_coupling=low is not None and low > settings.alpha,
        off_target_failures=tuple(sorted({r.perturbation for r in off_fails})),
    )


def off_target_rate(
    off: Sequence[CalibrationRow], rule: Rule, tau: float | None, settings: CalibrationSettings
) -> Rate:
    """Several perturbed clips share a scene: the interval is clustered by scene."""
    flags = [(r.scene_id, rule.fails(r, tau)) for r in off]
    return clustered_rate(flags, settings.bootstrap, settings.seed, settings.ci)


def tpr_by_level(
    positives: Sequence[CalibrationRow], rule: Rule, tau: float | None, ci: float
) -> tuple[LevelTpr, ...]:
    levels: list[LevelTpr] = []
    for level in sorted({r.pert_level for r in positives}):
        members = [r for r in positives if r.pert_level == level and r.has_event]
        verdicts = [rule.verdict(r, tau) for r in members]
        detected = sum(v is Verdict.FAIL for v in verdicts)
        decided = sum(v in DECIDED for v in verdicts)
        levels.append(
            LevelTpr(
                level=level,
                mu_median=median_or_none([r.mu for r in members if r.mu is not None]),
                total=rate(detected, len(members), ci),
                decided=rate(detected, decided, ci),
            )
        )
    return tuple(levels)


@dataclass(frozen=True)
class Detection:
    """One positive clip: injected magnitude, whether it was detected, its scene."""

    scene_id: str
    mu: float
    detected: bool
    level: int


def detections(
    positives: Sequence[CalibrationRow], rule: Rule, tau: float | None
) -> tuple[Detection, ...]:
    return tuple(
        Detection(r.scene_id, r.mu, rule.fails(r, tau), r.pert_level)
        for r in positives
        if r.has_event and r.mu is not None and r.mu > 0.0
    )


def detection_limit(
    found: Sequence[Detection], unit: str, settings: CalibrationSettings
) -> DetectionLimit:
    """L_D = exp((logit(1 - beta) - b0) / b1) from P(d | mu) = expit(b0 + b1 log mu)."""
    base = {"unit": unit, "beta": settings.beta, "n": len(found)}
    if not found:
        return DetectionLimit(
            value=None, ci=Ci(), censored="none", bound=None, reason="no_positives", **base
        )
    lowest = min(d.level for d in found)
    at_lowest = [d.detected for d in found if d.level == lowest]
    if np.mean(at_lowest) >= 1.0 - settings.beta:
        bound = min(d.mu for d in found)
        return DetectionLimit(value=None, ci=Ci(), censored="below", bound=bound, **base)
    value = ld_point(found, settings)
    if value is None:
        bound = max(d.mu for d in found)
        return DetectionLimit(value=None, ci=Ci(), censored="above", bound=bound, **base)
    ci, open_sides = ld_interval(found, settings)
    return DetectionLimit(
        value=value, ci=ci, censored="none", bound=None, ci_open=open_sides, **base
    )


def ld_interval(
    found: Sequence[Detection], settings: CalibrationSettings
) -> tuple[Ci, tuple[OpenSide, ...]]:
    """Profile-likelihood interval of L_D over the range the point estimate may take
    ([min mu, max mu] widened by (1 + ld_max_extrapolation) on each side). Unlike a scene
    bootstrap it drops no replicate whose fit is censored."""
    x = np.log([d.mu for d in found])
    widen = math.log1p(settings.ld_max_extrapolation)
    interval = profile_interval(
        x,
        np.asarray([d.detected for d in found], dtype=np.float64),
        _logit(1.0 - settings.beta),
        settings.platt_l2,
        settings.ci,
        (float(x.min()) - widen, float(x.max()) + widen),
    )
    candidates: tuple[tuple[OpenSide, float | None], ...] = (
        ("low", interval.low),
        ("high", interval.high),
    )
    sides = tuple(side for side, bound in candidates if bound is None)
    ci = Ci(
        low=None if interval.low is None else math.exp(interval.low),
        high=None if interval.high is None else math.exp(interval.high),
    )
    return ci, sides


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def ld_point(found: Sequence[Detection], settings: CalibrationSettings) -> float | None:
    """None when the fit has no increasing slope or extrapolates beyond the allowed range."""
    x = np.log([d.mu for d in found])
    y = np.asarray([d.detected for d in found], dtype=np.float64)
    fit = fit_logistic(x, y, l2=settings.platt_l2)
    if fit is None or fit.b <= 0.0:
        return None
    target = _logit(1.0 - settings.beta)
    value = math.exp(min((target - fit.a) / fit.b, 700.0))
    limit = (1.0 + settings.ld_max_extrapolation) * max(d.mu for d in found)
    return value if value <= limit else None
