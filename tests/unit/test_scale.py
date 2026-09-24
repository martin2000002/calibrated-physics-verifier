from __future__ import annotations

import math

import pytest
import records

from physverify.schema.errors import ScaleBindingError, SchemaError
from physverify.schema.scale import ScaleEstimate, ScaleSource, effective_scale


def test_none_requires_no_value() -> None:
    ScaleEstimate(meters_per_unit=None, sigma=math.nan, source=ScaleSource.NONE, lift_key="k")
    with pytest.raises(SchemaError):
        ScaleEstimate(meters_per_unit=1.0, sigma=math.nan, source=ScaleSource.NONE, lift_key="k")


@pytest.mark.parametrize("sigma", [math.nan, -0.1, math.inf])
def test_ground_truth_requires_finite_nonnegative_sigma(sigma: float) -> None:
    with pytest.raises(SchemaError):
        ScaleEstimate(
            meters_per_unit=1.0, sigma=sigma, source=ScaleSource.GROUND_TRUTH, lift_key="k"
        )
    ScaleEstimate(meters_per_unit=1.0, sigma=0.0, source=ScaleSource.GROUND_TRUTH, lift_key="k")


@pytest.mark.parametrize(
    "source", [ScaleSource.SIZE_PRIOR, ScaleSource.FREE_FALL, ScaleSource.METRIC_DEPTH]
)
def test_estimated_sources_accept_unknown_sigma(source: ScaleSource) -> None:
    estimate = ScaleEstimate(meters_per_unit=0.5, sigma=math.nan, source=source, lift_key="k")
    assert math.isnan(estimate.sigma)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, None])
def test_estimated_sources_need_a_positive_value(value: float | None) -> None:
    with pytest.raises(SchemaError):
        ScaleEstimate(meters_per_unit=value, sigma=0.1, source=ScaleSource.SIZE_PRIOR, lift_key="k")


def test_effective_scale() -> None:
    states = records.states()
    assert effective_scale(states, None) is None
    none = ScaleEstimate(
        meters_per_unit=None, sigma=math.nan, source=ScaleSource.NONE, lift_key=states.source_key
    )
    assert effective_scale(states, none) is None
    bound = ScaleEstimate(
        meters_per_unit=0.5, sigma=0.05, source=ScaleSource.FREE_FALL, lift_key=states.source_key
    )
    result = effective_scale(states, bound)
    assert result is not None
    assert result.meters_per_unit == 0.5 and result.rel_sigma == pytest.approx(0.1)


def test_effective_scale_refuses_another_table() -> None:
    estimate = ScaleEstimate(
        meters_per_unit=0.5, sigma=0.05, source=ScaleSource.FREE_FALL, lift_key="another-lift"
    )
    with pytest.raises(ScaleBindingError):
        effective_scale(records.states(), estimate)
