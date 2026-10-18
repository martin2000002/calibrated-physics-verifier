"""Metrics of the verifier, the reference VLM and the constant baselines on one policy view
(ADR 0011 D6-D8), with caption-clustered bootstrap intervals and the paired McNemar test."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from physverify.agreement.abstention import PolicyView, labels_of
from physverify.agreement.items import ComparisonItem
from physverify.agreement.stats import (
    Labels,
    Scores,
    accuracy,
    auc,
    balanced_accuracy,
    caption_bootstrap,
    classification,
    cohen_kappa,
    correlations,
    mcnemar_exact,
    mde_accuracy,
)

PAIRED_KEYS = ("accuracy", "balanced_accuracy", "kappa")


@dataclass(frozen=True)
class BootstrapPlan:
    n: int
    seed: int
    ci: float


def _scores(items: Sequence[ComparisonItem], index: Sequence[int], attr: str) -> Scores:
    return np.asarray([getattr(items[i], attr) for i in index], dtype=np.float64)


def _verifier_scores(items: Sequence[ComparisonItem], index: Sequence[int]) -> Scores:
    return np.asarray([items[i].decision.score for i in index], dtype=np.float64)


def system_metrics(
    y: Labels, p: Labels, score: Scores | None, human: Scores | None
) -> dict[str, float]:
    metrics = {"n": float(y.size), **classification(y, p)}
    metrics["predicted_implausible_rate"] = float(np.mean(p)) if p.size else math.nan
    if score is not None and human is not None:
        pearson, spearman = correlations(score, human)
        metrics.update({"auc": auc(y, score), "pearson": pearson, "spearman": spearman})
    return metrics


def constants(y: Labels, dev_majority: int | None) -> dict[str, dict[str, float]]:
    """Constant baselines: always implausible, always plausible, and the dev majority class."""
    table = {"implausible": 1, "plausible": 0}
    if dev_majority is not None:
        table["dev_majority"] = dev_majority
    return {
        name: system_metrics(y, np.full(y.size, v, dtype=np.int_), None, None)
        for name, v in table.items()
    }


def paired_statistic(y: Labels, a: Labels, b: Labels) -> Any:
    """Bootstrap statistic over item positions: each system's metrics and their differences."""

    def statistic(sample: Sequence[int]) -> Mapping[str, float]:
        pick = np.asarray(sample, dtype=np.int_)
        ys, pa, pb = y[pick], a[pick], b[pick]
        out = {}
        for key, metric in (
            ("accuracy", accuracy),
            ("balanced_accuracy", balanced_accuracy),
            ("kappa", cohen_kappa),
        ):
            va, vb = metric(ys, pa), metric(ys, pb)
            out.update({f"verifier.{key}": va, f"vlm.{key}": vb, f"difference.{key}": va - vb})
        return out

    return statistic


def evaluate_view(
    items: Sequence[ComparisonItem],
    view: PolicyView,
    dev_majority: int | None,
    plan: BootstrapPlan,
    test: tuple[float, float],
) -> dict[str, Any]:
    """``test`` is (alpha, power) for the minimum detectable difference."""
    y = labels_of(items, view.index, lambda it: it.human_implausible)
    human = _scores(items, view.index, "human_score")
    result: dict[str, Any] = {
        "policy": view.name,
        "paired": view.paired,
        "verifier": system_metrics(y, view.verifier, _verifier_scores(items, view.index), human),
        "constants": constants(y, dev_majority),
    }
    y_vlm = labels_of(items, view.vlm_index, lambda it: it.human_implausible)
    p_vlm = labels_of(items, view.vlm_index, lambda it: it.vlm_implausible)
    vlm_scores = _scores(items, view.vlm_index, "vlm_score")
    result["vlm"] = system_metrics(
        y_vlm, p_vlm, vlm_scores, _scores(items, view.vlm_index, "human_score")
    )
    if view.paired:
        result.update(_paired(items, view, y, p_vlm, plan, test))
    return result


def _paired(
    items: Sequence[ComparisonItem],
    view: PolicyView,
    y: Labels,
    p_vlm: Labels,
    plan: BootstrapPlan,
    test: tuple[float, float],
) -> dict[str, Any]:
    test_result = mcnemar_exact(y, view.verifier, p_vlm)
    discordant = (test_result.b + test_result.c) / y.size if y.size else math.nan
    positions = list(range(len(view.index)))
    captions = [items[i].row.caption for i in view.index]
    boot = caption_bootstrap(
        positions,
        lambda k: captions[k],
        paired_statistic(y, view.verifier, p_vlm),
        plan.n,
        plan.seed,
        plan.ci,
    )
    return {
        "mcnemar": {
            "b_verifier_only": test_result.b,
            "c_vlm_only": test_result.c,
            "p_value": test_result.p_value,
        },
        "discordant_rate": discordant,
        "mde_accuracy": mde_accuracy(y.size, discordant, *test),
        "bootstrap": {
            "replicas": boot.replicas,
            "dropped": boot.dropped,
            "non_finite": dict(boot.non_finite),
            "ci": plan.ci,
            "intervals": {k: [v.low, v.high] for k, v in boot.intervals.items()},
        },
    }


def h2_verdict(primary: Mapping[str, Any], calibrated: bool, alpha: float) -> dict[str, Any]:
    """H2 holds when McNemar p < alpha with the verifier more accurate AND the lower CI bound of
    kappa_verifier - kappa_vlm is above 0. Only the calibrated rule can support or reject it."""
    mcnemar = primary.get("mcnemar", {})
    p_value = float(mcnemar.get("p_value", math.nan))
    acc_gain = primary["verifier"]["accuracy"] - primary["vlm"]["accuracy"]
    interval = (
        primary.get("bootstrap", {})
        .get("intervals", {})
        .get("difference.kappa", [math.nan, math.nan])
    )
    holds = bool(p_value < alpha and acc_gain > 0 and interval[0] > 0)
    return {
        "applicable": calibrated,
        "supported": holds if calibrated else None,
        "reason": "calibrated rule"
        if calibrated
        else "pipeline test, no H2 verdict (provisional rule)",
        "mcnemar_p": p_value,
        "accuracy_difference": acc_gain,
        "kappa_difference_ci": list(interval),
        "mde_accuracy": primary.get("mde_accuracy", math.nan),
    }


def best_cut(y: Labels, score: Scores) -> float | None:
    """The cut on ``score`` (implausible when score > cut, NaN = plausible) with the highest
    balanced accuracy; ties go to the smaller cut. None without finite scores."""
    finite = np.unique(score[np.isfinite(score)])
    if not finite.size:
        return None
    candidates = [float(finite[0]) - 1.0, *map(float, finite)]
    best, best_value = None, -math.inf
    for cut in candidates:
        value = balanced_accuracy(y, predict_cut(score, cut))
        if math.isfinite(value) and value > best_value:
            best, best_value = cut, value
    return best


def predict_cut(score: Scores, cut: float) -> Labels:
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(score) & (score > cut), 1, 0).astype(np.int_)


def cut_metrics(
    items: Sequence[ComparisonItem], index: Sequence[int], cut: float
) -> dict[str, float]:
    """A secondary rule: implausible when the verifier score exceeds a fixed cut."""
    y = labels_of(items, index, lambda it: it.human_implausible)
    return {
        "cut": cut,
        **system_metrics(y, predict_cut(_verifier_scores(items, index), cut), None, None),
    }
