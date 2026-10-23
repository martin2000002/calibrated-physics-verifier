"""Decision threshold L_C (D5, Juicio 1), honest FPR (D6), TPR per level and L_D (D7, Juicio 3)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from physverify.calibration.artifact import DetectionLimit, Fpr, LevelTpr, OpenSide
from physverify.calibration.settings import CalibrationSettings
from physverify.calibration.stats import (
    clustered_rate,
    conformal_quantile,
    fit_logistic,
    median_or_none,
    profile_interval,
)
from physverify.calibration.table import (
    CalibrationRow,
    Role,
    corrected,
    require_one_clean_per_scene,
)
from physverify.calibration.verdict import DECIDED, Rule, clip_verdict
from physverify.schema.checks import Verdict
from physverify.stats.folds import grouped_folds
from physverify.stats.intervals import Ci, Rate, clopper_pearson_lower, rate

NOT_DETERMINED = "not_determined"
"""L_D is neither bounded below (too few clips bound the lowest-level rate above 1 - beta) nor
estimated (the fit fails or extrapolates below the tested range)."""


def verdict_of(row: CalibrationRow, rule: Rule, tau: float | None) -> Verdict:
    """The clip verdict of ``row``'s check, judged on its subjects (``clip_verdict``)."""
    return clip_verdict(row.subjects, rule, tau).verdict


def fails(row: CalibrationRow, rule: Rule, tau: float | None) -> bool:
    return verdict_of(row, rule, tau) is Verdict.FAIL


@dataclass(frozen=True)
class Tau:
    """``value`` = max(q, tolerance); ``conformal`` = q; ``n`` the ranked clean scenes and
    ``n_dropped`` the NaN magnitudes left out of the rank."""

    value: float | None
    sufficient: bool
    n: int
    conformal: float | None = None
    n_dropped: int = 0


def fit_tau(clean: Sequence[CalibrationRow], bias: float, alpha: float, tolerance: float) -> Tau:
    """Conformal quantile q over one clean clip per scene; tau = max(q, tolerance). The tolerance
    keeps a q made of floating-point noise (oracle clean ~1e-14) from becoming the threshold.

    q ranks m_max, the NaN-ignoring max magnitude over the clip's subjects (``subject_max``),
    while ``clip_verdict`` judges each subject. The coverage still holds: a subject i FAILs only
    if its m_i is a number and m_i - b - k sigma_i > tau, so m_max - b >= m_i - b > tau, and
    P(clean clip FAILs) <= P(m_max - b > tau) <= alpha. A clip is dropped from the rank (counted
    in ``n_dropped``) only when no subject has a numeric magnitude, and then none can FAIL."""
    require_one_clean_per_scene(clean)
    values = [corrected(row, bias) for row in clean if row.scored]
    found = conformal_quantile(values, alpha)
    value = None if found.value is None else max(found.value, tolerance)
    return Tau(value, found.sufficient, found.n, found.value, found.n_dropped)


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
            found[row.clip_id] = verdict_of(row, rule, tau)
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
    off_fails = [r for r in off if fails(r, rule, tau)]
    cv_total, cv_decided = false_positive_rates(list(cv.values()), settings.ci)
    in_total, in_decided = false_positive_rates(
        [verdict_of(r, rule, tau) for r in clean], settings.ci
    )
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
    flags = [(r.scene_id, fails(r, rule, tau)) for r in off]
    return clustered_rate(flags, settings.bootstrap, settings.seed, settings.ci)


def tpr_by_level(
    positives: Sequence[CalibrationRow], rule: Rule, tau: float | None, ci: float
) -> tuple[LevelTpr, ...]:
    levels: list[LevelTpr] = []
    for level in sorted({r.pert_level for r in positives}):
        members = [r for r in positives if r.pert_level == level and r.has_event]
        verdicts = [verdict_of(r, rule, tau) for r in members]
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
        Detection(r.scene_id, r.mu, fails(r, rule, tau), r.pert_level)
        for r in positives
        if r.has_event and r.mu is not None and r.mu > 0.0
    )


def detection_limit(
    found: Sequence[Detection], unit: str, settings: CalibrationSettings
) -> DetectionLimit:
    """L_D = exp((logit(1 - beta) - b0) / b1) from P(d | mu) = expit(b0 + b1 log mu).

    Censored below only when the one-sided Clopper-Pearson lower bound of the detection rate at
    the lowest level reaches 1 - beta (5/5 detected bounds it at about 0.55, which shows nothing).
    When the fit fails although the lowest level has some detection, or extrapolates below the
    tested range, L_D is not determined; ``above`` is left for a failed fit with nothing
    detected at the lowest level."""
    base = {"unit": unit, "beta": settings.beta, "n": len(found)}
    if not found:
        return DetectionLimit(
            value=None, ci=Ci(), censored="none", bound=None, reason="no_positives", **base
        )
    lowest = min(d.level for d in found)
    at_lowest = [d.detected for d in found if d.level == lowest]
    detected, mu_min = sum(at_lowest), min(d.mu for d in found)
    if clopper_pearson_lower(detected, len(at_lowest), settings.ci) >= 1.0 - settings.beta:
        return DetectionLimit(value=None, ci=Ci(), censored="below", bound=mu_min, **base)
    value = ld_point(found, settings)
    below_range = value is not None and value < mu_min / (1.0 + settings.ld_max_extrapolation)
    if (value is None and detected > 0) or below_range:
        return DetectionLimit(
            value=None, ci=Ci(), censored="none", bound=None, reason=NOT_DETERMINED, **base
        )
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
