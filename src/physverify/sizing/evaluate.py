"""Held-out error of the size model against baselines, with intervals by video (ADR 0010 §2.3).

r = log(L_hat / L). B0 is the frozen prior every unknown label gets today; B1 is the per-class
log-normal without the conformal factor (sigma_c as fitted)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from physverify.config.models import SizePrior
from physverify.sizing.model import SizeModelTable, class_prior
from physverify.sizing.samples import SizeSample, log_lengths
from physverify.stats.bootstrap import SceneBlock, scene_bootstrap
from physverify.stats.intervals import Ci

FloatArray = NDArray[np.float64]
MODEL, B0, B1 = "model", "b0_frozen_prior", "b1_unit_factor"
METRICS = ("median_abs_rel", "rms_log", "bias_log", "cover_1s", "cover_2s")


@dataclass(frozen=True)
class Errors:
    """Per-sample log error and predicted sigma (in log space) of one method."""

    log_error: FloatArray
    sigma: FloatArray


def errors_of(samples: Sequence[SizeSample], prior_of: Callable[[str], SizePrior]) -> Errors:
    priors = [prior_of(s.label) for s in samples]
    mean = np.log(np.array([p.mean_m for p in priors], dtype=np.float64))
    return Errors(
        log_error=mean - log_lengths(samples),
        sigma=np.array([p.rel_spread for p in priors], dtype=np.float64),
    )


def method_errors(
    samples: Sequence[SizeSample], model: SizeModelTable, baseline: SizePrior
) -> dict[str, Errors]:
    classes = model.classes
    return {
        MODEL: errors_of(samples, lambda label: class_prior(classes[label], model.sigma_factor)),
        B1: errors_of(samples, lambda label: class_prior(classes[label], 1.0)),
        B0: errors_of(samples, lambda _label: baseline),
    }


def metrics(errors: Errors) -> dict[str, float]:
    r, s = errors.log_error, errors.sigma
    if r.size == 0:
        return dict.fromkeys(METRICS, float("nan"))
    return {
        "median_abs_rel": float(np.median(np.abs(np.expm1(r)))),
        "rms_log": float(np.sqrt(np.mean(r**2))),
        "bias_log": float(np.mean(r)),
        "cover_1s": float(np.mean(np.abs(r) <= s)),
        "cover_2s": float(np.mean(np.abs(r) <= 2 * s)),
    }


def subset(errors: Errors, index: NDArray[np.intp]) -> Errors:
    return Errors(errors.log_error[index], errors.sigma[index])


def grouped_intervals(
    groups: Sequence[str],
    statistic: Callable[[NDArray[np.intp]], Mapping[str, float]],
    *,
    n_boot: int,
    seed: int,
    ci: float,
) -> dict[str, Ci]:
    """Percentile intervals of ``statistic(sample indices)`` over resamples of whole groups."""
    blocks = [SceneBlock(group, i) for i, group in enumerate(groups)]
    return scene_bootstrap(
        blocks, lambda index: statistic(np.asarray(index, dtype=np.intp)), n_boot, seed, ci
    )


def with_intervals(point: Mapping[str, float], bounds: Mapping[str, Ci]) -> dict[str, Any]:
    return {key: {"value": value, "ci": _bounds(bounds.get(key))} for key, value in point.items()}


def _bounds(ci: Ci | None) -> tuple[float | None, float | None]:
    return (None, None) if ci is None else (ci.low, ci.high)


def summary(
    errors: Mapping[str, Errors], groups: Sequence[str], *, n_boot: int, seed: int, ci: float
) -> dict[str, Any]:
    """Point metrics and percentile intervals per method, plus rms differences against MODEL,
    every key on the same video draws."""

    def statistic(index: NDArray[np.intp]) -> dict[str, float]:
        return _flat({name: subset(method, index) for name, method in errors.items()})

    flat = with_intervals(
        _flat(errors), grouped_intervals(groups, statistic, n_boot=n_boot, seed=seed, ci=ci)
    )
    out: dict[str, Any] = {name: {key: flat[f"{name}.{key}"] for key in METRICS} for name in errors}
    for baseline in (B0, B1):
        key = _difference_key(baseline)
        out[key] = flat[key]
    return out


def _flat(errors: Mapping[str, Errors]) -> dict[str, float]:
    out = {
        f"{name}.{key}": value
        for name, method in errors.items()
        for key, value in metrics(method).items()
    }
    for baseline in (B0, B1):
        out[_difference_key(baseline)] = _rms(errors[baseline]) - _rms(errors[MODEL])
    return out


def _difference_key(baseline: str) -> str:
    return f"rms_log_{baseline}_minus_model"


def _rms(errors: Errors) -> float:
    return float(np.sqrt(np.mean(errors.log_error**2))) if errors.log_error.size else float("nan")


def evaluate(
    samples: Sequence[SizeSample],
    model: SizeModelTable,
    baseline: SizePrior,
    *,
    n_boot: int,
    seed: int,
    ci: float,
) -> dict[str, Any]:
    """Totals with intervals by video, point metrics per class; samples of unknown classes are
    counted and left out."""
    known = [s for s in samples if s.label in model.classes]
    errors = method_errors(known, model, baseline)
    labels = np.array([s.label for s in known])
    per_class = {
        label: {
            "n": int((labels == label).sum()),
            **{
                name: metrics(subset(e, np.flatnonzero(labels == label)))
                for name, e in errors.items()
            },
        }
        for label in sorted(set(labels.tolist()))
    }
    return {
        "estimated": True,
        "n_objects": len(known),
        "n_videos": len({s.video_id for s in known}),
        "n_unknown_class": len(samples) - len(known),
        "sigma_factor": model.sigma_factor,
        "total": summary(errors, [s.video_id for s in known], n_boot=n_boot, seed=seed, ci=ci),
        "per_class": per_class,
    }
