"""The oracle of the analytic fixture gives the same results with the new defaults as with the old
behaviour behind each parameter.

The oracle never goes through the perceived CAMERA or LIFT, so the fixture's oracle results equal
those written by the code of commit 01140e2. R4_jump and R4_size run as v1 (``v1_checks``): the
registry now holds their v2, compared in ``test_v2_checks_oracle.py``."""

from __future__ import annotations

import json

from golden import same
from oracle_golden_case import GOLDEN, v1_checks
from oracle_golden_case import oracle_results as golden_case_results
from synthetic_clip import SyntheticFixtureDataset

from physverify.app.plugins import PLUGIN_KINDS
from physverify.schema.checks import CheckResult
from physverify.verification.check_set import ChecksSettings, build_checks
from physverify.verification.inputs import OracleSource
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationRunner
from physverify.verification.thresholding import ThresholdTable

OLD = {
    "R4_missing": {"sigma_model": "unknown"},
    "R5a": {"travel_noise": "declared"},
    "R5b": {"travel_noise": "declared"},
}


def oracle_results(overrides: dict[str, dict[str, str]]) -> tuple[CheckResult, ...]:
    dataset = SyntheticFixtureDataset()
    checks = build_checks(PLUGIN_KINDS["checks"](), ChecksSettings(overrides))
    runner = VerificationRunner(checks, RawPolicy(), ThresholdTable.provisional({}))
    report = runner.run(OracleSource(dataset), dataset.clips())
    return report.results


def test_the_oracle_results_do_not_move() -> None:
    new, old = oracle_results({}), oracle_results(OLD)
    assert [r.check_id for r in new if r.check_id.startswith(("R4", "R5"))]
    assert [r.to_dict() for r in new] == [r.to_dict() for r in old]


def test_the_v1_oracle_results_match_the_golden() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(golden_case_results(*v1_checks()), sort_keys=True))
    assert len(golden) == len(current) > 0
    assert same(current, golden)
