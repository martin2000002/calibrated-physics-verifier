"""scale/relative: relative scale error and coverage."""

from __future__ import annotations

import math

import numpy as np
import pytest
from metric_cases import METRIC_OPTIONS as OPTIONS
from metric_cases import metric_case
from synthetic_clip import manifest

from physverify.diagnosis.measurement_cases import MetricOutcome
from physverify.diagnosis.metrics.scale_relative import ScaleRelativeMetric, source_key
from physverify.schema.scale import ScaleEstimate, ScaleSource

LIFT_KEY = "lift:fixture"


def truth() -> ScaleEstimate:
    return ScaleEstimate(
        meters_per_unit=2.0, sigma=0.0, source=ScaleSource.GROUND_TRUTH, lift_key=LIFT_KEY
    )


def estimate(meters_per_unit: float | None, source: ScaleSource) -> ScaleEstimate:
    return ScaleEstimate(
        meters_per_unit=meters_per_unit, sigma=math.nan, source=source, lift_key=LIFT_KEY
    )


def measure(pred: ScaleEstimate) -> MetricOutcome:
    case = metric_case(pred, truth())
    return ScaleRelativeMetric(OPTIONS).samples(case)


def test_ten_percent_high_scale() -> None:
    outcome = measure(estimate(2.2, ScaleSource.SIZE_PRIOR))
    np.testing.assert_allclose(outcome.samples["scale_rel"], [0.1], atol=1e-12)
    np.testing.assert_allclose(outcome.samples["scale_rel/size_prior"], [0.1], atol=1e-12)
    assert outcome.ratios["coverage"] == (1.0, 1.0)


def test_no_scale_gives_no_sample_and_no_coverage() -> None:
    outcome = measure(estimate(None, ScaleSource.NONE))
    assert outcome.samples["scale_rel"].size == 0
    assert outcome.samples["scale_rel/none"].size == 0
    assert outcome.ratios["coverage"] == (0.0, 1.0)


def test_sample_key_names_the_source() -> None:
    assert source_key(estimate(1.0, ScaleSource.FREE_FALL)) == "scale_rel/free_fall"


def test_compute_summarizes_without_context() -> None:
    summary = ScaleRelativeMetric(OPTIONS).compute(
        estimate(1.5, ScaleSource.METRIC_DEPTH), truth(), manifest()
    )
    assert summary["scale_rel.mean"] == pytest.approx(-0.25)
    assert summary["coverage"] == 1.0
