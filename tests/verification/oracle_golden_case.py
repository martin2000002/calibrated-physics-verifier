"""Oracle check results of the analytic fixture, written once by the code before it2 (commit
01140e2) into ``golden/oracle_before_it2.json`` and compared with the current code (it2 T14).
The same module runs in both trees:

    PYTHONPATH=<old>/src:tests/fixtures python tests/verification/oracle_golden_case.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from synthetic_clip import SyntheticFixtureDataset

from physverify.cli.context import PLUGIN_KINDS
from physverify.schema.io import json_ready
from physverify.verification.inputs import OracleSource
from physverify.verification.policies.raw import RawPolicy
from physverify.verification.runner import ChecksSettings, VerificationRunner, build_checks

GOLDEN = Path(__file__).resolve().parent / "golden" / "oracle_before_it2.json"


def oracle_results() -> list[Any]:
    dataset = SyntheticFixtureDataset()
    checks = build_checks(PLUGIN_KINDS["checks"](), ChecksSettings({}))
    report = VerificationRunner(checks, RawPolicy(), {}).run(OracleSource(dataset), dataset.clips())
    return [json_ready(r.to_dict()) for r in report.results]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    out.write_text(json.dumps(oracle_results(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
