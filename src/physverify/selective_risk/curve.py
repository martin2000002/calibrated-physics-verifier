"""Risk-coverage of the threshold policy as k_sigma sweeps, and the choice of k* (ADR 0010 §3).

The rule is never copied: every point of the curve is ``ThresholdPolicy(k_sigma=k).decide`` on
the same perceived results. The grid is set in alpha (one-sided miss probability of a Gaussian
interval), k = Phi^-1(1 - alpha); both are reported. A case is one reference event: its
reference verdict and the perceived result paired with it (None when perception missed it,
which counts as an abstention)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from physverify.calibration.table import Role
from physverify.core.errors import MissingInputError
from physverify.diagnosis.pairing import Paired
from physverify.schema.checks import CheckResult, InputLevel, Verdict
from physverify.selective_risk.settings import AbstentionSettings
from physverify.stats.intervals import clopper_pearson
from physverify.verification.policies.threshold import ThresholdParams, ThresholdPolicy

ABSTAINED, PASSED, FAILED = 0, 1, 2
_CODES = {Verdict.PASS: PASSED, Verdict.FAIL: FAILED}
REFERENCE_K = 0.0
"""The reference verdict is the point decision (no interval) of the oracle magnitude."""
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Case:
    clip_id: str
    check_id: str
    reference: Verdict
    perceived: CheckResult | None


@dataclass(frozen=True)
class CaseCounts:
    """Reference events dropped before the sweep, by why."""

    no_reference: int = 0
    spurious: int = 0

    def __add__(self, other: CaseCounts) -> CaseCounts:
        return CaseCounts(self.no_reference + other.no_reference, self.spurious + other.spurious)


def alpha_grid(settings: AbstentionSettings) -> NDArray[np.float64]:
    return np.geomspace(settings.alpha_max, settings.alpha_min, settings.grid_points)


def k_of(alpha: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(norm.isf(alpha), dtype=np.float64)


# --- References -------------------------------------------------------------------------------


def point_reference(oracle: CheckResult, thresholds: Mapping[str, Any]) -> Verdict | None:
    """The oracle's point verdict against ``thresholds`` (None if it cannot be decided); callers
    pass ``raw_table`` so that the reference never carries a perceived-level correction."""
    policy = ThresholdPolicy(ThresholdParams(k_sigma=REFERENCE_K, on_unknown_sigma="point"))
    verdict = policy.decide(oracle, thresholds).verdict
    return verdict if verdict in _CODES else None


def raw_table(thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """The same taus with no calibration correction (bias 0, sigma factor 1)."""
    checks = {
        check_id: {"tau": float(entry["tau"])}
        for check_id, entry in thresholds.get("checks", {}).items()
    }
    return {**thresholds, "checks": checks, "artifact": None}


def paired_cases(
    paired: Paired, reference_table: Mapping[str, Any]
) -> tuple[list[Case], CaseCounts]:
    """One case per oracle event with a point reference against ``raw_table(reference_table)``
    (the oracle level's taus); an event the perceived chain missed is a case with no perceived
    result."""
    reference = raw_table(reference_table)
    cases: list[Case] = []
    undecidable = 0
    events = [(p.oracle, p.other) for p in paired.pairs] + [(r, None) for r in paired.missing]
    for oracle, perceived in events:
        verdict = point_reference(oracle, reference)
        if verdict is None:
            undecidable += 1
            continue
        cases.append(Case(oracle.clip_id, oracle.check_id, verdict, perceived))
    return cases, CaseCounts(no_reference=undecidable, spurious=len(paired.spurious))


def require_taus(table: Mapping[str, Any], level: InputLevel) -> None:
    """Refuse a table with no tau for any check: every event would drop or abstain."""
    checks = table.get("checks", {})
    if not any(entry.get("tau") is not None for entry in checks.values()):
        raise MissingInputError(
            f"the {level.value} threshold table ({table.get('source')}) has no tau for any "
            "check; pass a table with --oracle-table/--perceived-table or use --provisional"
        )


def injected_reference(role: Role) -> Verdict | None:
    """A8's injected label of a (clip, check) as a verdict: clean PASS, injected FAIL; coupled
    and off-target checks have no label."""
    return {Role.CLEAN: Verdict.PASS, Role.POSITIVE: Verdict.FAIL}.get(role)


# --- Sweep ------------------------------------------------------------------------------------


def decision_codes(
    cases: Sequence[Case], ks: Iterable[float], thresholds: Mapping[str, Any]
) -> IntArray:
    """``(len(ks), len(cases))`` codes of ``ThresholdPolicy(k).decide`` on each perceived result."""
    rows = []
    for k in ks:
        policy = ThresholdPolicy(ThresholdParams(k_sigma=float(k)))
        rows.append([_code(policy, case.perceived, thresholds) for case in cases])
    return np.array(rows, dtype=np.int64).reshape(-1, len(cases))


def _code(
    policy: ThresholdPolicy, result: CheckResult | None, thresholds: Mapping[str, Any]
) -> int:
    if result is None:
        return ABSTAINED
    return _CODES.get(policy.decide(result, thresholds).verdict, ABSTAINED)


def row_codes(keys: Sequence[tuple[str, str]], codes: IntArray) -> dict[tuple[str, str], int]:
    """Codes of each (clip, check) row by the rule of ``verify``'s table: FAIL if any event
    fails, PASS if all pass, else abstained."""
    rows: dict[tuple[str, str], set[int]] = {}
    for key, code in zip(keys, codes.tolist(), strict=True):
        rows.setdefault(key, set()).add(int(code))
    return {
        key: FAILED if FAILED in seen else PASSED if seen == {PASSED} else ABSTAINED
        for key, seen in rows.items()
    }


def reference_codes(cases: Sequence[Case]) -> IntArray:
    return np.array([_CODES[c.reference] for c in cases], dtype=np.int64)


@dataclass(frozen=True)
class CurvePoint:
    n: int
    decided: int
    errors: int
    false_fail: int
    """FAIL decided on a PASS reference."""
    ref_pass_decided: int
    missed_fail: int
    """PASS decided on a FAIL reference."""
    ref_fail_decided: int

    @property
    def coverage(self) -> float:
        return self.decided / self.n if self.n else float("nan")

    @property
    def risk(self) -> float:
        return self.errors / self.decided if self.decided else float("nan")


def curve_point(codes: IntArray, reference: IntArray) -> CurvePoint:
    decided = codes != ABSTAINED
    wrong = decided & (codes != reference)
    on_pass, on_fail = decided & (reference == PASSED), decided & (reference == FAILED)
    return CurvePoint(
        n=int(codes.size),
        decided=int(decided.sum()),
        errors=int(wrong.sum()),
        false_fail=int((on_pass & (codes == FAILED)).sum()),
        ref_pass_decided=int(on_pass.sum()),
        missed_fail=int((on_fail & (codes == PASSED)).sum()),
        ref_fail_decided=int(on_fail.sum()),
    )


def risk_bounds(errors: int, decided: int, ci: float) -> tuple[float, float]:
    """Clopper-Pearson bounds of a risk; (0, 1) when nothing was decided."""
    bound = clopper_pearson(errors, decided, ci)
    return (
        0.0 if bound.low is None else bound.low,
        1.0 if bound.high is None else bound.high,
    )


def aurc(coverage: Sequence[float], risk: Sequence[float]) -> float:
    """Area under risk(coverage) by trapezoids over the decided points, divided by the
    coverage span (a mean risk); NaN without a decided point."""
    points = sorted((c, r) for c, r in zip(coverage, risk, strict=True) if np.isfinite(r) and c > 0)
    if not points:
        return float("nan")
    cov = np.array([p[0] for p in points])
    rsk = np.array([p[1] for p in points])
    span = float(cov[-1] - cov[0])
    if span == 0.0:
        return float(rsk.mean())
    return float(np.trapezoid(rsk, cov) / span)


def curve_aurc(codes: IntArray, reference: IntArray) -> float:
    points = [curve_point(row, reference) for row in codes]
    return aurc([p.coverage for p in points], [p.risk for p in points])
