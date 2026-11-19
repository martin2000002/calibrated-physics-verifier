"""Clips left without input, per set and check: the gap a device-keyed cache leaves silently."""

from __future__ import annotations

from dataclasses import replace

from calibration_cases import CHECKS, config, synthetic_report

from physverify.calibration.collect import level_data
from physverify.calibration.input_gaps import all_input_gaps, input_gaps
from physverify.schema.checks import AbstainReason, CheckResult, InputLevel, Verdict
from physverify.verification.runner import INPUT_GAP_KEY


def _gap(result: CheckResult, *, runner: bool = True) -> CheckResult:
    details = {**result.details, INPUT_GAP_KEY: True} if runner else result.details
    reason = AbstainReason.MISSING_INPUT
    return replace(result, verdict=Verdict.ABSTAIN, abstain_reason=reason, details=details)


def test_a_check_whose_every_clip_lacks_input_counts_them_all() -> None:
    data = level_data(InputLevel.ORACLE, synthetic_report(10), CHECKS, config(), split=None)
    target = data.results[0].check_id
    gapped = replace(
        data,
        results=tuple(_gap(r) if r.check_id == target else r for r in data.results),
    )
    found = {g.check_id: (g.missing, g.n) for g in input_gaps("set", gapped)}
    n = found[target][1]
    assert n > 0 and found[target] == (n, n)
    assert all(m == 0 for c, (m, _) in found.items() if c != target)
    assert {g.dataset for g in all_input_gaps("set", data, {"real": gapped})} == {"set", "real"}


def test_a_clip_with_one_result_that_got_input_is_not_a_gap() -> None:
    data = level_data(InputLevel.ORACLE, synthetic_report(10), CHECKS, config(), split=None)
    first = data.results[0]
    one = replace(data, results=(_gap(first), first))
    assert [(g.missing, g.n) for g in input_gaps("set", one)] == [(0, 1)]


def test_a_checks_own_missing_input_on_a_scene_is_not_a_gap() -> None:
    """R3a without a contact, R2 without a finite gravity: the check ran on its input."""
    data = level_data(InputLevel.ORACLE, synthetic_report(10), CHECKS, config(), split=None)
    own = replace(data, results=tuple(_gap(r, runner=False) for r in data.results))
    assert all(g.missing == 0 for g in input_gaps("set", own))
