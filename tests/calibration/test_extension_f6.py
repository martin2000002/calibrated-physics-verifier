"""The known-size extension (ADR 0053, subset declared at render time, ``checks_only = ["R5b"]``):
R5b alone reads f6, every other check fits and dev-scores bit-identically with or without it."""

from __future__ import annotations

import pytest
from calibration_cases import CHECKS, config, extension_report, fit_meta, merged
from calibration_cases import synthetic_report as base_report

from physverify.calibration.collect import level_data
from physverify.calibration.dev_score import dev_score
from physverify.calibration.extension import ExtensionScope
from physverify.calibration.fit import fit_checks
from physverify.schema.checks import InputLevel
from physverify.schema.clip import Split
from physverify.synth.plan import family_of

CONFIG = config(alpha=0.1, folds=3, bootstrap=30)
SCOPED = "R5b"


@pytest.mark.parametrize("level", [InputLevel.ORACLE, InputLevel.PERCEIVED])
def test_every_other_check_is_bit_identical_with_calib_f6(level: InputLevel) -> None:
    scoped = SCOPED
    base, extra = base_report(n_scenes=18, level=level), extension_report(level=level)
    scope = ExtensionScope({c.scene_id or "": frozenset({scoped}) for c in extra.clips})
    without = level_data(level, base, CHECKS, CONFIG, split=Split.CALIB)
    with_ext = level_data(
        level, merged(base, extra), CHECKS, CONFIG, split=Split.CALIB, scope=scope
    )
    others = [c.check_id for c in CHECKS if c.check_id != scoped]
    assert others
    before, after = fit_checks(without, CHECKS, CONFIG), fit_checks(with_ext, CHECKS, CONFIG)
    assert all(before[c].model_dump_json() == after[c].model_dump_json() for c in others)
    found = [
        dev_score((d,), CHECKS, fit_meta(), CONFIG, family_of).report.levels[0].checks
        for d in (without, with_ext)
    ]
    assert all(found[0][c].model_dump_json() == found[1][c].model_dump_json() for c in others)
    assert found[0][scoped].model_dump_json() != found[1][scoped].model_dump_json()
