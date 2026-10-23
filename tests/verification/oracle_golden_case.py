"""Oracle check results of the analytic fixture, written once by the code before it2 (commit
01140e2) into ``golden/oracle.json`` and compared with the current code (it2 T14).
The same module runs in both trees:

    PYTHONPATH=<old>/src:tests/fixtures python tests/verification/oracle_golden_case.py

Since it4 the registry holds R4_jump v2 and R4_size v2, so the comparisons with the golden pass
``v1_checks()`` (the preregistered R4_jump and R4_size "self"); the other seven checks run as
registered.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from synthetic_clip import SyntheticFixtureDataset

from physverify.app.plugins import PLUGIN_KINDS
from physverify.checks._base import Check
from physverify.schema.io import json_ready
from physverify.verification.check_set import ChecksSettings, build_checks
from physverify.verification.inputs import OracleSource
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import VerificationRunner

GOLDEN = Path(__file__).resolve().parent / "golden" / "oracle.json"


V1_OVERRIDES: dict[str, dict[str, Any]] = {"R4_size": {"reference": "self"}}


def v1_checks() -> tuple[dict[str, type[Check]], dict[str, dict[str, Any]]]:
    """The registry with the v1 R4_jump, and the overrides that make R4_size v1 (it4 J4, J5)."""
    from reference.r4_jump_v1 import ContinuityCheck

    return {**PLUGIN_KINDS["checks"](), "R4_jump": ContinuityCheck}, V1_OVERRIDES


def oracle_results(
    registry: Mapping[str, type[Check]] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> list[Any]:
    dataset = SyntheticFixtureDataset()
    found = PLUGIN_KINDS["checks"]() if registry is None else registry
    checks = build_checks(found, ChecksSettings(overrides or {}))
    report = VerificationRunner(checks, RawPolicy(), {}).run(OracleSource(dataset), dataset.clips())
    return [json_ready(r.to_dict()) for r in report.results]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    out.write_text(json.dumps(oracle_results(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
