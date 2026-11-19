"""Clips a check never got its input for, counted per set from the evaluated results: only the
runner's own abstentions (``INPUT_GAP_KEY``), never a check's ``MISSING_INPUT`` on a scene."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from physverify.calibration.fit import LevelData
from physverify.calibration.gap_record import InputGap
from physverify.verification.runner import INPUT_GAP_KEY


def input_gaps(dataset: str, data: LevelData) -> tuple[InputGap, ...]:
    """Per check, the clips whose every result is a runner input gap, out of its clips."""
    missing: dict[tuple[str, str], bool] = {}
    for r in data.results:
        key = (r.check_id, r.clip_id)
        missing[key] = missing.get(key, True) and r.details.get(INPUT_GAP_KEY) is True
    per_check: dict[str, list[bool]] = defaultdict(list)
    for (check_id, _), flag in missing.items():
        per_check[check_id].append(flag)
    return tuple(
        InputGap(dataset=dataset, level=data.level, check_id=c, missing=sum(f), n=len(f))
        for c, f in sorted(per_check.items())
    )


def all_input_gaps(
    dataset: str, synthetic: LevelData, real: Mapping[str, LevelData]
) -> tuple[InputGap, ...]:
    return (
        *input_gaps(dataset, synthetic),
        *(gap for name, data in sorted(real.items()) for gap in input_gaps(name, data)),
    )
