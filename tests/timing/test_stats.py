from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from physverify.core.errors import TimingError
from physverify.timing.records import Sample
from physverify.timing.stats import (
    ACROSS_CLIPS,
    across_clips,
    per_clip,
    quantile,
    stat_rows,
    summarize,
)

MAX_IQR_REL = 0.10


def sample(clip: str, value: float, *, measure: str = "x.wall_s", warmup: bool = False) -> Sample:
    return Sample("t", clip, 0, warmup, measure, "s", value)


def test_quantile_matches_numpy_linear_on_random_vectors() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        values = sorted(rng.exponential(3.0, size=int(rng.integers(1, 40))).tolist())
        for p in (0.0, 0.25, 0.5, 0.75, 1.0, float(rng.uniform())):
            expected = float(np.quantile(values, p, method="linear"))
            assert quantile(values, p) == pytest.approx(expected, abs=1e-12, rel=0)


def test_summary_of_one_to_four() -> None:
    summary = summarize([4.0, 1.0, 3.0, 2.0])
    assert (summary.median, summary.p25, summary.p75) == (2.5, 1.75, 3.25)
    assert summary.iqr == 1.5
    assert summary.iqr_rel == pytest.approx(0.6)
    assert summary.mean == 2.5
    assert summary.sd == pytest.approx(float(np.std([1, 2, 3, 4], ddof=1)))
    assert (summary.min, summary.max, summary.n) == (1.0, 4.0, 4)


def test_one_sample_has_nan_spread() -> None:
    summary = summarize([5.0])
    assert (summary.median, summary.p25, summary.p75, summary.iqr) == (5.0, 5.0, 5.0, 0.0)
    assert math.isnan(summary.sd)
    assert summary.iqr_rel == 0.0


def test_zero_median_gives_nan_relative_iqr() -> None:
    assert math.isnan(summarize([0.0, 0.0, 0.0]).iqr_rel)


@pytest.mark.parametrize("call", [lambda: summarize([]), lambda: quantile([], 0.5)])
def test_empty_samples_are_an_error(call: Callable[[], object]) -> None:
    with pytest.raises(TimingError):
        call()


def test_quantile_level_outside_unit_interval_is_an_error() -> None:
    with pytest.raises(TimingError):
        quantile([1.0], 1.5)


def test_warmup_samples_never_enter_statistics() -> None:
    rows = per_clip([sample("a", 100.0, warmup=True), sample("a", 1.0), sample("a", 3.0)], 0.1)
    (row,) = rows
    assert (row.n, row.median, row.max) == (2, 2.0, 3.0)


def test_across_clip_rows_use_per_clip_medians() -> None:
    samples = [sample("a", v) for v in (1.0, 2.0, 3.0)] + [sample("b", v) for v in (10.0, 30.0)]
    clip_rows = per_clip(samples, MAX_IQR_REL)
    (across,) = across_clips(clip_rows, MAX_IQR_REL)
    assert across.clip_id == ACROSS_CLIPS
    assert across.n == 2
    assert across.median == pytest.approx((2.0 + 20.0) / 2)
    assert across.unstable


def test_stat_rows_list_clips_first_then_across() -> None:
    samples = [sample("a", 1.0), sample("b", 1.0), sample("a", 2.0, measure="y.wall_s")]
    rows = stat_rows(samples, MAX_IQR_REL)
    assert [(r.clip_id, r.measure) for r in rows] == [
        ("a", "x.wall_s"),
        ("b", "x.wall_s"),
        ("a", "y.wall_s"),
        (ACROSS_CLIPS, "x.wall_s"),
        (ACROSS_CLIPS, "y.wall_s"),
    ]
    assert not rows[0].unstable
