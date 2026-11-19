"""The choice of k*, the paired AURC difference and the report rows of the curve."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import norm

from physverify.selective_risk.curve import (
    Case,
    CurvePoint,
    IntArray,
    curve_aurc,
    curve_point,
    decision_codes,
    reference_codes,
    risk_bounds,
)
from physverify.selective_risk.records import AurcDifference, PointRow
from physverify.stats.bootstrap import ClusterBlock, scene_bootstrap
from physverify.verification.thresholding import ThresholdTable


def choose_k(
    points: Sequence[CurvePoint], ks: Sequence[float], risk_target: float, ci: float
) -> int | None:
    """Index of the largest coverage whose risk upper bound is <= ``risk_target`` (ties: the
    smallest k); None when no point qualifies."""
    best: int | None = None
    for i, point in enumerate(points):
        if point.decided == 0 or risk_bounds(point.errors, point.decided, ci)[1] > risk_target:
            continue
        if best is None or _better(point, ks[i], points[best], ks[best]):
            best = i
    return best


def _better(point: CurvePoint, k: float, other: CurvePoint, other_k: float) -> bool:
    if point.coverage != other.coverage:
        return point.coverage > other.coverage
    return k < other_k


def paired_aurc_difference(
    first: IntArray,
    second: IntArray,
    reference: IntArray,
    clip_ids: Sequence[str],
    *,
    n_boot: int,
    seed: int,
    ci: float,
) -> AurcDifference:
    """AURC(second) - AURC(first) with a percentile interval; each replica resamples whole
    clips once and scores both tables on that same draw."""

    def difference(index: Sequence[int]) -> dict[str, float]:
        d = np.asarray(index, dtype=np.intp)
        return {
            "diff": curve_aurc(second[:, d], reference[d]) - curve_aurc(first[:, d], reference[d])
        }

    blocks = [ClusterBlock(clip_id, i) for i, clip_id in enumerate(clip_ids)]
    bound = scene_bootstrap(blocks, difference, n_boot, seed, ci).get("diff")
    return AurcDifference(
        value=curve_aurc(second, reference) - curve_aurc(first, reference),
        ci=(None, None) if bound is None else (bound.low, bound.high),
        n_boot_non_finite=None if bound is None else bound.non_finite,
    )


def point_row(point: CurvePoint, k: float, alpha: float, ci: float) -> PointRow:
    return PointRow(
        k_sigma=k,
        alpha=alpha,
        n=point.n,
        decided=point.decided,
        coverage=point.coverage,
        risk=point.risk,
        risk_ci=risk_bounds(point.errors, point.decided, ci),
        false_fail_rate=_ratio(point.false_fail, point.ref_pass_decided),
        missed_fail_rate=_ratio(point.missed_fail, point.ref_fail_decided),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def curve_rows(
    codes: IntArray, reference: IntArray, ks: Sequence[float], alphas: Sequence[float], ci: float
) -> list[PointRow]:
    return [
        point_row(curve_point(row, reference), k, a, ci)
        for row, k, a in zip(codes, ks, alphas, strict=True)
    ]


def per_check_rows(
    cases: Sequence[Case],
    thresholds: ThresholdTable,
    ks: Mapping[str, float],
    ci: float,
) -> dict[str, dict[str, PointRow]]:
    """Risk and coverage of each check at the named k values (e.g. k* and k = 2)."""
    out: dict[str, dict[str, PointRow]] = {}
    for check_id in sorted({c.check_id for c in cases}):
        mine = [c for c in cases if c.check_id == check_id]
        reference = reference_codes(mine)
        codes = decision_codes(mine, ks.values(), thresholds)
        out[check_id] = {
            name: point_row(curve_point(row, reference), k, float(norm.sf(k)), ci)
            for (name, k), row in zip(ks.items(), codes, strict=True)
        }
    return out


def z_scores(cases: Sequence[Case], thresholds: ThresholdTable) -> list[float]:
    """(m - bias - tau) / (factor * sigma) of each perceived result (report only, never stored
    in ``details``); NaN when it is undefined."""
    out = []
    for case in cases:
        result = case.perceived
        entry = None if result is None else thresholds.entry_of(result.check_id)
        if result is None or entry is None or result.magnitude is None:
            out.append(float("nan"))
            continue
        scale = float("nan") if entry.sigma_factor is None else entry.sigma_factor * result.sigma
        out.append((result.magnitude - entry.bias - entry.tau) / scale if scale else float("nan"))
    return out
