"""Complementarity of the reference VLM and the verifier on hand-built comparison items."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.datasets._base import LabelledVideo
from physverify.human_agreement.clip_decision import ClipDecision, Decision
from physverify.human_agreement.comparison_items import ComparisonItem
from physverify.human_agreement.complementarity import (
    SHARES,
    complementarity,
    flags,
    statistic,
)
from physverify.schema.clip import Split

IMPL, PLAUS = ClipDecision.IMPLAUSIBLE, ClipDecision.PLAUSIBLE
ABSTAIN, NONE = ClipDecision.ABSTAIN, ClipDecision.NO_EVIDENCE

# (caption, human implausible, VLM implausible or None, verifier outcome)
FIXTURE = (
    ("c1", True, True, IMPL),
    ("c1", True, False, IMPL),
    ("c2", True, True, ABSTAIN),
    ("c2", True, False, NONE),
    ("c3", True, True, PLAUS),
    ("c3", False, False, PLAUS),
    ("c4", False, False, IMPL),
    ("c4", False, False, ABSTAIN),
    ("c5", False, True, ABSTAIN),
    ("c5", True, None, IMPL),
)


def _item(k: int, caption: str, human: bool, vlm: bool | None, outcome: ClipDecision):
    row = LabelledVideo(
        part="p",
        version="v1",
        split=Split.ALL,
        clip_id=f"clip{k}",
        video_url=f"u{k}",
        caption=caption,
        generator="g",
        human_raw=0.0 if human else 1.0,
        human_cut=0.5,
        reference_raw=math.nan if vlm is None else (0.0 if vlm else 1.0),
        reference_cut=0.5,
        reference_name="vlm",
        violated_rules="",
    )
    decision = Decision(row.clip_id, outcome, math.nan, None, ())
    return ComparisonItem(row, human, vlm, decision, outcome)


def _items(spec):
    return [_item(k, *entry) for k, entry in enumerate(spec)]


def test_the_fixture_matches_the_hand_computation() -> None:
    """Clip 10 has no VLM prediction and is left out: n = 9 paired clips, positives 1-5 (P = 5).

    VLM flags 1, 3, 5, 9 (4), true 1, 3, 5: coverage 3/5, precision 3/4, flagged 4/9.
    Verifier flags only IMPLAUSIBLE (as_plausible): 1, 2, 7 (3), true 1, 2: coverage 2/5,
    precision 2/3, flagged 3/9. Either flags 1, 2, 3, 5, 7, 9 (6), true 1, 2, 3, 5: coverage 4/5,
    precision 4/6, flagged 6/9.
    Shares of the 5 positives: verifier only {2} 1/5, VLM only {3, 5} 2/5, both {1} 1/5,
    neither {4} 1/5.
    Decided (plausible or implausible): 1, 2, 5, 6, 7, fraction 5/9. On them the verifier flags
    1, 2, 7 (precision 2/3) and the VLM 1, 5 (precision 2/2 = 1), difference 2/3 - 1 = -1/3.
    """
    result = complementarity(_items(FIXTURE), 0, 0, 0.95)
    expected = {
        "vlm.coverage": 3 / 5,
        "vlm.precision": 3 / 4,
        "vlm.flagged_rate": 4 / 9,
        "verifier.coverage": 2 / 5,
        "verifier.precision": 2 / 3,
        "verifier.flagged_rate": 3 / 9,
        "either.coverage": 4 / 5,
        "either.precision": 4 / 6,
        "either.flagged_rate": 6 / 9,
        "share.verifier_only": 1 / 5,
        "share.vlm_only": 2 / 5,
        "share.both": 1 / 5,
        "share.neither": 1 / 5,
        "decided.fraction": 5 / 9,
        "decided.verifier.precision": 2 / 3,
        "decided.vlm.precision": 1.0,
        "decided.difference.precision": -1 / 3,
    }
    assert result.point == pytest.approx(expected)
    assert result.counts == {
        "n": 9,
        "human_implausible": 5,
        "decided": 5,
        "flagged.vlm": 4,
        "flagged.verifier": 3,
        "flagged.either": 6,
    }
    assert (result.flag_policy, result.decided_policy) == ("as_plausible", "decided_only")


def test_the_shares_of_any_resample_sum_to_one() -> None:
    f = flags(_items(FIXTURE))
    rng = np.random.default_rng(3)
    for _ in range(20):
        sample = rng.integers(0, f.human.size, size=f.human.size).tolist()
        values = statistic(f, sample)
        if f.human[sample].any():
            assert sum(values[f"share.{name}"] for name in SHARES) == pytest.approx(1.0)


def test_the_caption_bootstrap_is_seeded_and_covers_every_key() -> None:
    items = _items(FIXTURE)
    first = complementarity(items, 200, 20261017, 0.95)
    again = complementarity(items, 200, 20261017, 0.95)
    assert first == again
    assert first.replicas == 200 and set(first.intervals) == set(first.point)
    low, high = first.intervals["either.coverage"]
    assert 0.0 <= low <= first.point["either.coverage"] <= high <= 1.0


def test_without_decided_clips_the_verifier_flags_nothing_and_its_precision_is_nan() -> None:
    """Every clip abstains: the verifier flags 0 (coverage 0/2, precision 0/0 = NaN), either is
    the VLM (flags clips 1 and 3, true 1: coverage 1/2, precision 1/2), the 2 positives are
    VLM only (clip 1) and neither (clip 2), decided fraction 0/4, precisions on decided NaN in
    the point and in every replica."""
    spec = (
        ("c1", True, True, ABSTAIN),
        ("c1", True, False, NONE),
        ("c2", False, True, ABSTAIN),
        ("c2", False, False, NONE),
    )
    result = complementarity(_items(spec), 50, 1, 0.95)
    point = result.point
    assert point["verifier.coverage"] == 0.0 and math.isnan(point["verifier.precision"])
    assert point["either.coverage"] == point["vlm.coverage"] == 0.5
    assert point["either.precision"] == point["vlm.precision"] == 0.5
    assert [point[f"share.{n}"] for n in SHARES] == [0.0, 0.5, 0.0, 0.5]
    assert point["decided.fraction"] == 0.0
    assert math.isnan(point["decided.verifier.precision"])
    assert math.isnan(point["decided.difference.precision"])
    assert result.non_finite["decided.verifier.precision"] == result.replicas == 50
    assert all(math.isnan(v) for v in result.intervals["decided.vlm.precision"])


def test_without_paired_clips_every_ratio_is_nan_and_nothing_is_resampled() -> None:
    result = complementarity(_items((("c1", True, None, IMPL),)), 50, 1, 0.95)
    assert result.counts["n"] == 0 and result.replicas == 0 and result.intervals == {}
    assert all(math.isnan(v) for v in result.point.values())
