"""it3 T11: the oracle and V8 never go through the perceived CONTACTS provider.

The fixture's oracle results equal the golden of it2 (``oracle_before_it2.json``, byte-identical
to what 2a762ad writes), and its V8 contacts (every stage ground truth) equal those written by
2a762ad, also when the plan asks for ``contacts/kinematic``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

from support import SettingsFactory
from truth_contacts_golden_case import GOLDEN, truth_contacts

from physverify.extraction._base import GROUND_TRUTH, StageProvider
from physverify.extraction.plan import resolve_plan
from physverify.schema.contacts import ContactTable
from physverify.schema.stages import STAGE_ORDER, Stage

VERIFICATION = str(Path(__file__).resolve().parents[1] / "verification")
if VERIFICATION not in sys.path:
    sys.path.insert(0, VERIFICATION)

import oracle_golden_case  # noqa: E402


def test_t11_the_oracle_results_do_not_move() -> None:
    golden = json.loads(oracle_golden_case.GOLDEN.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(oracle_golden_case.oracle_results(), sort_keys=True))
    assert len(golden) == len(current) > 0
    assert current == golden


def test_t11_the_v8_contacts_do_not_move() -> None:
    assert truth_contacts() == ContactTable.load_json(GOLDEN)


def test_t11_v8_uses_ground_truth_contacts_whatever_the_plan(
    make_settings: SettingsFactory, real_providers: Mapping[str, type[StageProvider]]
) -> None:
    settings = make_settings("extraction.plan.contacts=kinematic", profile="precise")
    plan = resolve_plan(
        settings, real_providers, dataset="synthetic_fixture", gt_stages=STAGE_ORDER
    )
    assert plan.step(Stage.CONTACTS).name == GROUND_TRUTH
    perceived = resolve_plan(settings, real_providers, dataset="synthetic_fixture")
    assert perceived.step(Stage.CONTACTS).name == "kinematic"
