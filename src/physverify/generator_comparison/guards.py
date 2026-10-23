"""Guards and strata of the pre-registered generator comparison."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from physverify.core.models import FrozenModel
from physverify.generator_comparison.analysis_inputs import (
    AnalysisInputs,
    ArmSummary,
    Values,
    as_diffs,
)
from physverify.generator_comparison.arms import PromptArms
from physverify.generator_comparison.settings import ComparisonSettings
from physverify.generator_comparison.stats import TestResult, mean_interval, paired_test, supported
from physverify.generator_comparison.verdict import GuardState

ABSTENTION = "FT abstention rate minus base"
MOTION = "FT median motion against (1 - max_motion_drop) x base median"
OBJECTS = "base objects_ok rate minus FT"


class GuardResult(FrozenModel):
    state: GuardState
    value: float | None = None
    limit: float | None = None
    detail: str = ""


def _limit_guard(
    value: float | None, limit: float, ok: Callable[[float, float], bool], detail: str
) -> GuardResult:
    if value is None:
        return GuardResult(state="not_evaluated", limit=limit, detail=detail)
    state: GuardState = "ok" if ok(value, limit) else "fail"
    return GuardResult(state=state, value=value, limit=limit, detail=detail)


def signal_guards(
    ft: ArmSummary, base: ArmSummary, inputs: AnalysisInputs
) -> dict[str, GuardResult]:
    """G1-G3: abstention rise, motion drop and objects_ok drop of FT against base."""
    sel, cfg = inputs.select, inputs.cfg
    rise = _minus(ft.abstention, base.abstention)
    drop = _minus(base.objects_ok_rate, ft.objects_ok_rate)
    floor = None if base.median_motion is None else (1 - sel.max_motion_drop) * base.median_motion
    motion = None if floor is None else ft.median_motion
    return {
        "G1": _limit_guard(rise, sel.max_abstention_rise, lambda v, lim: v <= lim, ABSTENTION),
        "G2": _limit_guard(motion, floor or 0.0, lambda v, lim: v >= lim, MOTION),
        "G3": _limit_guard(drop, cfg.max_objects_ok_drop, lambda v, lim: v <= lim, OBJECTS),
    }


def _minus(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def sa_guard(found: tuple[Values, int], inputs: AnalysisInputs) -> GuardResult:
    margin = inputs.cfg.sa_guard_margin
    if inputs.external_status != "ok":
        return GuardResult(state="not_evaluated", limit=-margin, detail="external judge not ok")
    _, hi = mean_interval(as_diffs(found[0]), inputs.cfg)
    if hi is None:
        return GuardResult(state="not_evaluated", limit=-margin, detail="too few SA pairs")
    state: GuardState = "fail" if hi < -margin else "ok"
    return GuardResult(state=state, value=hi, limit=-margin, detail="upper CI of mean SA(ft - bo8)")


def missing_pair_guard(
    primary: TestResult, worst: TestResult, cfg: ComparisonSettings
) -> GuardResult:
    if not supported(primary, cfg.alpha):
        return GuardResult(state="not_evaluated", detail="primary not supported")
    if worst.hl is None:
        return GuardResult(state="not_evaluated", detail="no imputable prompt")
    state: GuardState = "fail" if worst.hl <= 0 else "ok"
    return GuardResult(
        state=state, value=worst.hl, limit=0.0, detail="HL with missing pairs as S=0"
    )


def strata(
    values: Values, arms: Sequence[PromptArms], cfg: ComparisonSettings
) -> dict[str, TestResult]:
    groups: dict[str, list[str]] = {"unseen": [], "seen": []}
    for arm in arms:
        groups["unseen" if arm.unseen else "seen"].append(arm.prompt_id)
        groups.setdefault(f"family:{arm.prompt.family}", []).append(arm.prompt_id)
    return {
        name: paired_test(as_diffs({p: values[p] for p in ids if p in values}), cfg)
        for name, ids in sorted(groups.items())
    }
