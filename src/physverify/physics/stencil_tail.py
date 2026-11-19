"""The maximum of a per-stencil statistic over a clip's stencils, restated as the maximum over a
reference number of stencils with the same family-wise tail."""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.special import erfcx, gammainc, gammaincinv

STENCIL_ROWS = 4
"""Rows of one third-difference stencil."""
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def stencils_in(rows: int) -> int:
    """Third-difference stencils of a table of ``rows`` rows."""
    return max(rows - STENCIL_ROWS + 1, 0)


def chi3_log_sf(m: float) -> float:
    """log P(chi_3 > m), the tail of the norm of three independent standard normals.

    P(chi_3 > m) = 2 (Phi(-m) + m phi(m)) = 2 phi(m) (m + R(m)), with the Mills ratio
    R(m) = Phi(-m) / phi(m) = sqrt(pi / 2) erfcx(m / sqrt(2)), finite in log form for any m >= 0."""
    mills = math.sqrt(math.pi / 2.0) * float(erfcx(m / math.sqrt(2.0)))
    return math.log(2.0) - 0.5 * m * m - _LOG_SQRT_2PI + math.log(m + mills)


def reference_max(magnitude: float, stencils: int, reference_stencils: int) -> float:
    """m_ref such that the maximum of ``reference_stencils`` independent chi_3 draws exceeds m_ref
    as often as the maximum of ``stencils`` draws exceeds ``magnitude``.

    With F the chi_3 CDF, the family-wise tail of m over n stencils is p = 1 - F(m)^n, and
    m_ref = F^-1((1 - p)^(1 / n_ref)) = F^-1(F(m)^(n / n_ref)). The identity when n = n_ref (the
    value is returned untouched); NaN and m <= 0 are returned as they are."""
    if stencils == reference_stencils or not (math.isfinite(magnitude) and magnitude > 0.0):
        return magnitude
    if stencils < 1 or reference_stencils < 1:
        raise ValueError(f"stencil counts must be >= 1, got {stencils} and {reference_stencils}")
    ratio = stencils / reference_stencils
    if _chi3_cdf(magnitude) <= 0.5:
        return _chi3_ppf(_chi3_cdf(magnitude) ** ratio)
    target = _reference_log_sf(chi3_log_sf(magnitude), ratio)
    return _invert_log_sf(target, magnitude)


def _chi3_cdf(m: float) -> float:
    """P(chi_3 <= m) = P(3/2, m^2 / 2), the regularized lower incomplete gamma."""
    return float(gammainc(1.5, 0.5 * m * m))


def _chi3_ppf(q: float) -> float:
    return math.sqrt(2.0 * float(gammaincinv(1.5, q)))


def _reference_log_sf(log_sf: float, ratio: float) -> float:
    """log(1 - F(m)^ratio) from log(1 - F(m)) <= log(1/2); log_sf + log(ratio) once the tail
    underflows."""
    tail = math.exp(log_sf)
    if tail == 0.0:
        return log_sf + math.log(ratio)
    return math.log(-math.expm1(ratio * math.log1p(-tail)))


def _invert_log_sf(target: float, start: float) -> float:
    """The m >= 0 with chi3_log_sf(m) = target < 0 (``chi3_log_sf`` decreases from 0 at m = 0)."""
    high = start
    while chi3_log_sf(high) > target:
        high = 2.0 * high + 1.0
    low = 0.0 if chi3_log_sf(start) <= target else start
    return float(brentq(lambda m: chi3_log_sf(m) - target, low, high, xtol=1e-14, rtol=1e-15))
