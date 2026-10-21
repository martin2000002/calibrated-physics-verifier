"""it2 T14: the oracle never goes through the perceived CAMERA or LIFT, so the fixture's oracle
results equal those written by the code before it2 (01140e2)."""

from __future__ import annotations

import json

from golden import same
from oracle_golden_case import GOLDEN, oracle_results


def test_t14_the_oracle_results_do_not_move() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(oracle_results(), sort_keys=True))
    assert len(golden) == len(current) > 0
    assert same(current, golden)
