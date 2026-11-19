import math

import numpy as np
import pytest
from scipy.stats import chi

from physverify.physics.stencil_tail import chi3_log_sf, reference_max, stencils_in


def test_a_track_of_t_rows_has_t_minus_3_stencils() -> None:
    assert [stencils_in(rows) for rows in (0, 3, 4, 24, 48)] == [0, 0, 1, 21, 45]


@pytest.mark.parametrize("m", [0.1, 1.0, 3.78, 8.0, 30.0])
def test_the_tail_is_the_chi3_tail(m: float) -> None:
    assert chi3_log_sf(m) == pytest.approx(float(chi.logsf(m, 3)), rel=1e-9)


@pytest.mark.parametrize("m", [0.0, 1.5, 3.78, 12.0])
def test_the_reference_count_is_the_identity(m: float) -> None:
    assert reference_max(m, 21, 21) == m


@pytest.mark.parametrize("m", [math.nan, math.inf, 0.0, -1.0])
def test_values_without_a_tail_are_kept(m: float) -> None:
    out = reference_max(m, 45, 21)
    assert out == m or (math.isnan(out) and math.isnan(m))


@pytest.mark.parametrize("stencils", [7, 45, 117])
def test_the_family_wise_tail_is_kept(stencils: int) -> None:
    """P(max of n > m) = 1 - F(m)^n is the same at the reference count."""
    m = 3.5
    restated = reference_max(m, stencils, 21)
    tail = 1.0 - chi.cdf(m, 3) ** stencils
    assert 1.0 - chi.cdf(restated, 3) ** 21 == pytest.approx(tail, rel=1e-9)


def test_more_stencils_lower_the_value_and_fewer_raise_it() -> None:
    assert reference_max(3.78, 45, 21) < 3.78 < reference_max(3.78, 7, 21)
    values = [reference_max(m, 45, 21) for m in np.linspace(0.5, 40.0, 50)]
    assert np.all(np.diff(values) > 0)


def test_far_tails_stay_finite() -> None:
    assert reference_max(200.0, 117, 21) == pytest.approx(200.0, rel=1e-2)


def test_the_maximum_of_white_noise_is_restated_to_the_reference_quantile() -> None:
    """The 95 % quantile of the maximum over 45 chi_3 norms maps to the one over 21."""
    q45 = float(chi.ppf(0.95 ** (1 / 45), 3))
    q21 = float(chi.ppf(0.95 ** (1 / 21), 3))
    assert reference_max(q45, 45, 21) == pytest.approx(q21, rel=1e-9)


def test_counts_below_one_are_refused() -> None:
    with pytest.raises(ValueError, match="stencil counts"):
        reference_max(3.0, 0, 21)


@pytest.mark.parametrize("m", [1e-6, 0.3, 1.0, 1.6, 2.5])
def test_values_near_zero_keep_the_family_wise_tail(m: float) -> None:
    for stencils in (7, 45):
        restated = reference_max(m, stencils, 21)
        expected = float(chi.ppf(chi.cdf(m, 3) ** (stencils / 21), 3))
        assert restated == pytest.approx(expected, rel=1e-8)
