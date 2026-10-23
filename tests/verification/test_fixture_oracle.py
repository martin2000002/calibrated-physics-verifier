"""AC13: the analytic fixture clip through the oracle and the nine real checks."""

from __future__ import annotations

import pytest
from synthetic_clip import SyntheticFixtureDataset

from physverify.app.plugins import PLUGIN_KINDS
from physverify.schema.checks import CheckResult
from physverify.verification.inputs import OracleSource
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import (
    ChecksSettings,
    RowStatus,
    TableRow,
    VerificationReport,
    VerificationRunner,
    build_checks,
)

BOUNDS = {
    "R1": 1e-9,
    "R2": 1e-7,
    "R4_missing": 0.0,
    "R4_size": 1e-12,
    "R5a": 1e-9,
    "R5b": 1e-9,
}
NO_EVENT = ("R3a", "R3b")


@pytest.fixture(scope="module")
def report() -> VerificationReport:
    dataset = SyntheticFixtureDataset()
    checks = build_checks(PLUGIN_KINDS["checks"](), ChecksSettings({}))
    return VerificationRunner(checks, RawPolicy(), {}).run(OracleSource(dataset), dataset.clips())


def rows(report: VerificationReport) -> dict[str, TableRow]:
    return {row.check_id: row for row in report.table}


@pytest.mark.parametrize(("check_id", "bound"), sorted(BOUNDS.items()))
def test_the_oracle_magnitudes_are_null(
    report: VerificationReport, check_id: str, bound: float
) -> None:
    row = rows(report)[check_id]
    assert row.status is RowStatus.SCORED
    assert row.magnitude is not None and abs(row.magnitude) <= bound


@pytest.mark.parametrize("check_id", NO_EVENT)
def test_the_collision_checks_see_no_event(report: VerificationReport, check_id: str) -> None:
    assert rows(report)[check_id].status is RowStatus.NO_EVENT


def test_every_check_has_a_row_and_nothing_abstains(report: VerificationReport) -> None:
    assert set(rows(report)) == set(PLUGIN_KINDS["checks"]())
    assert all(result.abstain_reason is None for result in report.results)


def scored(report: VerificationReport) -> list[CheckResult]:
    return [result for result in report.results if result.magnitude is not None]


def test_the_oracle_has_zero_sigma(report: VerificationReport) -> None:
    assert scored(report)
    assert all(result.sigma == 0.0 for result in scored(report))


def test_every_frame_span_is_inside_the_clip(report: VerificationReport) -> None:
    last = SyntheticFixtureDataset().clips()[0].n_frames - 1
    spans = [r.frame_span for r in report.results if r.frame_span is not None]
    assert spans
    assert all(0 <= start <= end <= last for start, end in spans)
