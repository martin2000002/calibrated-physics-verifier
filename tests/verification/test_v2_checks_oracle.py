"""The v2 registry on the analytic fixture, and the versioning of the two magnitudes.

The seven checks left unchanged give, with the v2 registry, the oracle results of the golden
written by commit 01140e2; R4_jump and R4_size v2 are new versions whose parameters hash
differently, so no calibration, exclusion record or scope written for v1 is accepted for v2."""

from __future__ import annotations

import json
from typing import Any

from golden import same
from oracle_golden_case import GOLDEN, V1_OVERRIDES, oracle_results, v1_checks

from physverify.app.plugins import PLUGIN_KINDS
from physverify.verification.check_set import ChecksSettings, checks_hash

CHANGED = ("R4_jump", "R4_size")


def unchanged(results: list[Any]) -> list[Any]:
    return [r for r in results if r["check_id"] not in CHANGED]


def test_ac1_the_seven_other_checks_do_not_move() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(oracle_results(), sort_keys=True))
    assert len(unchanged(current)) == len(unchanged(golden)) > 0
    assert same(unchanged(current), unchanged(golden))


def test_the_v2_results_carry_their_own_unit() -> None:
    current = oracle_results()
    units = {r["check_id"]: r["unit"] for r in current if r["check_id"] in CHANGED}
    assert units == {"R4_jump": "jump_over_noise", "R4_size": "rel_size_change"}


def test_v1_and_v2_hash_differently() -> None:
    registry, overrides = v1_checks()
    v1 = checks_hash(registry, ChecksSettings(overrides))
    v2 = checks_hash(PLUGIN_KINDS["checks"](), ChecksSettings({}))
    assert v1 != v2
    only_size = checks_hash(PLUGIN_KINDS["checks"](), ChecksSettings(V1_OVERRIDES))
    assert len({v1, v2, only_size}) == 3
