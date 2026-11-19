"""The coverage rows of the final table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from physverify.calibration.artifact import EvaluationReport
from physverify.reporting.final_rows import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    Criteria,
    Row,
    ci_text,
    fmt,
    missing,
    stratum_at,
)
from physverify.selective_risk.records import CurveRecord, PointRow
from physverify.stats.intervals import clopper_pearson


def best_point_risk(curve: Sequence[PointRow], target: float) -> PointRow | None:
    within = [p for p in curve if p.risk <= target]
    return max(within, key=lambda p: p.coverage, default=None)


def coverage_rows(
    curve: CurveRecord | None,
    kubric: EvaluationReport | None,
    preflight: Any,
    c: Criteria,
    srcs: Mapping[str, str],
    ci: float,
) -> list[Row]:
    return [
        *_curve_rows(curve, c, srcs["curve"], ci),
        _preflight_row(preflight),
        _kubric_floor_row(kubric, preflight, c, srcs["kubric"]),
    ]


def _curve_rows(curve: CurveRecord | None, c: Criteria, src: str, ci: float) -> list[Row]:
    texts = (
        ("T.curve", "risk-coverage curve reported", ">= 2 points"),
        (
            "T.rule",
            "frozen rule: max coverage with risk upper bound <= target",
            f"risk upper <= {c.risk_target}",
        ),
        (
            "T.point_risk",
            "secondary: max coverage with point risk <= target",
            f"risk <= {c.risk_target}",
        ),
    )
    if curve is None:
        return [missing(i, t, th, "abstention report") for i, t, th in texts]
    points = curve.tables.calibrated.curve
    best = curve.tables.calibrated.best
    rows = [
        Row(
            texts[0][0],
            texts[0][1],
            f"{len(points)} points",
            "",
            texts[0][2],
            PASS if len(points) >= 2 else FAIL,
            src,
        )
    ]
    if best is None:
        rows.append(Row(texts[1][0], texts[1][1], "n/a", "", texts[1][2], FAIL, src, "no k meets"))
    else:
        p = points[best]
        covered = clopper_pearson(p.decided, p.n, ci)
        rows.append(
            Row(
                texts[1][0],
                texts[1][1],
                f"coverage {fmt(p.coverage)}",
                f"coverage {ci_text(covered.low, covered.high)}",
                texts[1][2],
                PASS if p.risk_ci[1] <= c.risk_target else FAIL,
                src,
                f"risk {fmt(p.risk)} {ci_text(*p.risk_ci)}; k {p.k_sigma}, n {p.n}",
            )
        )
    second = best_point_risk(points, c.risk_target)
    rows.append(
        Row(
            texts[2][0],
            texts[2][1],
            fmt(None if second is None else second.coverage),
            "",
            texts[2][2],
            NOT_EVALUABLE if second is None else PASS,
            src,
            "" if second is None else f"k {second.k_sigma}, risk {fmt(second.risk)}",
        )
    )
    return rows


def _preflight_row(preflight: Any) -> Row:
    text = "per-check verdict; weight-0 checks never make a clip abstain"
    if preflight is None:
        return missing("T.active_set", text, "no weight-0 check changes a verdict", "preflight")
    found, src = preflight
    leaking = list(found.get("unweighted_abstentions", []))
    active = ", ".join(found.get("active_set", []))
    return Row(
        "T.active_set",
        text,
        f"{len(leaking)} leaking",
        "",
        "0 leaking",
        FAIL if leaking else PASS,
        src,
        f"active set: {active}",
    )


def _kubric_floor_row(
    kubric: EvaluationReport | None, preflight: Any, c: Criteria, src: str
) -> Row:
    text = "Kubric perceived: decided fraction per active check (clean held-out clips)"
    perceived = None if kubric is None else stratum_at(kubric, "perceived")
    if perceived is None or preflight is None:
        return missing("T.floor_kubric", text, f">= {c.kubric_decided}", "evaluation or preflight")
    active = set(preflight[0].get("active_set", []))
    shares = {
        h.check_id: h.fpr_clean_decided.n / h.fpr_clean.n
        for h in perceived.checks
        if h.check_id in active and h.fpr_clean.n
    }
    low = sorted(k for k, v in shares.items() if v < c.kubric_decided)
    note = "; ".join(f"{k} {fmt(v)}" for k, v in sorted(shares.items()))
    return Row(
        "T.floor_kubric",
        text,
        f"{len(shares) - len(low)}/{len(shares)}",
        "",
        f">= {c.kubric_decided}",
        FAIL if low or not shares else PASS,
        src,
        note + " (estimated)",
    )
