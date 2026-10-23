"""it1 (changes A and B): the oracle of the analytic fixture gives the same results with the new
defaults as with the old behaviour behind each parameter."""

from __future__ import annotations

from synthetic_clip import SyntheticFixtureDataset

from physverify.app.plugins import PLUGIN_KINDS
from physverify.schema.checks import CheckResult
from physverify.verification.inputs import OracleSource
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import ChecksSettings, VerificationRunner, build_checks

OLD = {
    "R4_missing": {"sigma_model": "unknown"},
    "R5a": {"travel_noise": "declared"},
    "R5b": {"travel_noise": "declared"},
}


def oracle_results(overrides: dict[str, dict[str, str]]) -> tuple[CheckResult, ...]:
    dataset = SyntheticFixtureDataset()
    checks = build_checks(PLUGIN_KINDS["checks"](), ChecksSettings(overrides))
    report = VerificationRunner(checks, RawPolicy(), {}).run(OracleSource(dataset), dataset.clips())
    return report.results


def test_the_oracle_results_do_not_move() -> None:
    new, old = oracle_results({}), oracle_results(OLD)
    assert [r.check_id for r in new if r.check_id.startswith(("R4", "R5"))]
    assert [r.to_dict() for r in new] == [r.to_dict() for r in old]
