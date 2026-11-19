"""Size-estimator metrics on hand-made errors and the video bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from physverify.config.models import SizePrior
from physverify.sizing.evaluate import B0, B1, MODEL, Errors, evaluate, grouped_intervals, metrics
from physverify.sizing.model import ClassSize, SizeModelTable
from physverify.sizing.samples import SizeSample


def test_metrics_on_known_errors() -> None:
    r = np.log([1.1, 0.8, 1.0, 2.0])
    got = metrics(Errors(r, np.full(4, 0.2)))
    assert got["median_abs_rel"] == pytest.approx(np.median([0.1, 0.2, 0.0, 1.0]))
    assert got["rms_log"] == pytest.approx(math.sqrt(np.mean(r**2)))
    assert got["cover_1s"] == pytest.approx(0.5)
    assert got["cover_2s"] == pytest.approx(0.75)


def test_draws_take_whole_videos() -> None:
    ids = ["a", "a", "b", "c", "c", "c"]
    seen: list[list[str]] = []

    def statistic(index: np.ndarray) -> dict[str, float]:
        seen.append([ids[i] for i in index])
        return {"n": float(index.size)}

    grouped_intervals(ids, statistic, n_boot=20, seed=0, ci=0.9)
    assert len(seen) == 20
    for picked in seen:
        for video in set(picked):
            assert picked.count(video) % ids.count(video) == 0


def test_the_report_is_marked_as_an_estimate() -> None:
    cup = ClassSize(mu_log=math.log(0.1), sigma_log=0.1, n_objects=3, n_videos=3)
    model = SizeModelTable(sigma_factor=2.0, classes={"cup": cup})
    samples = [SizeSample("cup", f"v{i}", (0.1 * 1.05**i,) * 3, "test") for i in range(6)]
    samples.append(SizeSample("bike", "b", (1.0,) * 3, "test"))
    report = evaluate(
        samples, model, SizePrior(mean_m=0.1, rel_spread=0.5), n_boot=50, seed=0, ci=0.9
    )
    assert report["estimated"] is True and report["n_unknown_class"] == 1
    total = report["total"]
    assert total[MODEL]["rms_log"]["value"] == pytest.approx(total[B1]["rms_log"]["value"])
    assert total[B0]["rms_log"]["value"] == pytest.approx(total[MODEL]["rms_log"]["value"])
    assert total[MODEL]["cover_1s"]["value"] >= total[B1]["cover_1s"]["value"]
