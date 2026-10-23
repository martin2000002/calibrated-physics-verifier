"""Check weights (informedness lower bound) and the violation probability (D11, Juicio 2)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from physverify.calibration.schema import Probability, Weight
from physverify.calibration.settings import CalibrationSettings
from physverify.calibration.stats import balanced_weights, ece, fit_logistic, mad
from physverify.stats.bootstrap import SceneBlock, scene_bootstrap
from physverify.stats.folds import grouped_folds
from physverify.stats.intervals import Ci

J_KEY = "informedness"
ECE_KEY = "ece"


@dataclass(frozen=True)
class Flag:
    """A positive (detected?) or a clean negative (false positive out of fold?)."""

    scene_id: str
    positive: bool
    flag: bool


def _informedness(flags: Sequence[Flag]) -> float:
    pos = [f.flag for f in flags if f.positive]
    neg = [f.flag for f in flags if not f.positive]
    if not pos or not neg:
        return math.nan
    return float(np.mean(pos)) - float(np.mean(neg))


def raw_weight(flags: Sequence[Flag], settings: CalibrationSettings) -> Weight:
    """J = TPR_total - FPR_cv; weight max(0, lower bound of J) before normalisation."""
    blocks = [SceneBlock(f.scene_id, f) for f in flags]
    ci = scene_bootstrap(
        blocks,
        lambda s: {J_KEY: _informedness(s)},
        settings.bootstrap,
        settings.seed,
        settings.ci,
    ).get(J_KEY, Ci())
    value = _informedness(flags)
    return Weight(
        value=max(0.0, ci.low) if ci.low is not None else 0.0,
        informedness=value if math.isfinite(value) else None,
        informedness_ci=ci,
        n_positive=sum(f.positive for f in flags),
        n_negative=sum(not f.positive for f in flags),
    )


def normalised(weights: Mapping[str, Weight]) -> dict[str, Weight]:
    """w / max w; all 0 when no check beats chance with confidence."""
    top = max((w.value for w in weights.values()), default=0.0)
    return {
        check_id: w.model_copy(update={"value": w.value / top if top > 0.0 else 0.0})
        for check_id, w in weights.items()
    }


@dataclass(frozen=True)
class Labelled:
    scene_id: str
    magnitude: float
    """m - bias."""
    positive: bool


def fit_probability(
    data: Sequence[Labelled], tau: float | None, settings: CalibrationSettings
) -> Probability:
    """Monotone class-balanced Platt fit on u = (m - tau) / s, s = MAD of the negatives."""
    scale = mad([d.magnitude for d in data if not d.positive])
    if tau is None:
        return Probability(a=None, b=None, scale=scale, ece_cv=None, ece_ci=Ci(), n=len(data))
    u, y = _design(data, tau, scale)
    fit = fit_logistic(u, y, l2=settings.platt_l2, weights=balanced_weights(y), monotone=True)
    if fit is None:
        return Probability(a=None, b=None, scale=scale, ece_cv=None, ece_ci=Ci(), n=len(data))
    ece_cv, ece_ci = _cross_fitted_ece(data, tau, scale, settings)
    return Probability(a=fit.a, b=fit.b, scale=scale, ece_cv=ece_cv, ece_ci=ece_ci, n=len(data))


def _design(data: Sequence[Labelled], tau: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray([(d.magnitude - tau) / scale for d in data], dtype=np.float64)
    y = np.asarray([d.positive for d in data], dtype=np.float64)
    return u, y


@dataclass(frozen=True)
class Prediction:
    p: float
    y: float
    weight: float


def _cross_fitted_ece(
    data: Sequence[Labelled], tau: float, scale: float, settings: CalibrationSettings
) -> tuple[float | None, Ci]:
    weights = balanced_weights([d.positive for d in data])
    blocks: list[SceneBlock[Prediction]] = []
    for fold in grouped_folds([d.scene_id for d in data], settings.folds, settings.seed):
        train = [d for d in data if d.scene_id not in fold]
        u, y = _design(train, tau, scale)
        fit = fit_logistic(u, y, l2=settings.platt_l2, weights=balanced_weights(y), monotone=True)
        if fit is None:
            continue
        for index, d in enumerate(data):
            if d.scene_id in fold:
                p = float(fit.predict([(d.magnitude - tau) / scale])[0])
                blocks.append(
                    SceneBlock(d.scene_id, Prediction(p, float(d.positive), weights[index]))
                )
    if not blocks:
        return None, Ci()
    value = _ece_of([b.item for b in blocks])
    ci = scene_bootstrap(
        blocks, lambda s: {ECE_KEY: _ece_of(s)}, settings.bootstrap, settings.seed, settings.ci
    )
    return (value if math.isfinite(value) else None), ci.get(ECE_KEY, Ci())


def _ece_of(predictions: Sequence[Prediction]) -> float:
    return ece(
        [p.p for p in predictions], [p.y for p in predictions], [p.weight for p in predictions]
    )
